# PhD Debate #19 Verdict: pinch_column_fraction Correction & Phase AE Cross-Device Validation

**Date**: 2026-02-27
**Moderator**: Claude (Opus 4.6)
**Panel**: Dr. PP (Pulsed Power), Dr. DPF (Dense Plasma Focus), Dr. EE (Electrical Engineering)

---

## VERDICT: MAJORITY (2-1)

### Score: 6.3/10 (DOWN from 6.6)

### Question

Assess the pinch_column_fraction (pcf) correction and the Phase AE cross-device validation framework. Does the pcf=0.14 fix for PF-1000, the CrossValidator infrastructure, and the published fc/fm benchmarking constitute meaningful progress toward experimental validation?

### Answer

The pinch_column_fraction correction is a physically motivated improvement that correctly reduces the PF-1000 current dip from 76% to ~33% (matching Scholz 2006 experimental data). However, pcf=0.14 is a fit parameter, not a derived quantity. The CrossValidator infrastructure is sound and was fixed during this debate (pcf propagation bug). The Phase AE test suite (22 tests) provides meaningful coverage of cross-device prediction and pcf sensitivity. Overall, this represents incremental infrastructure improvement but no new experimental validation or physics advancement. The score drops from 6.6 to 6.3, reflecting a more rigorous re-assessment that revealed the D2 molecular mass density error, confirmed the 21% timing error as model-form, and found no post-D1 calibration run has been performed.

---

## Consensus Verification Checklist

- [x] **Mathematical derivation provided** -- Circuit EOM: V_cap = L_total * dI/dt + I * dL/dt + R_total * I. Snowplow: F_rad = (mu_0/4pi) * (fc*I)^2 * z_f/r_s. L_radial = (mu_0/2pi) * z_f * ln(b/r_eff). All derived from Maxwell's equations and momentum conservation.
- [x] **Dimensional analysis verified** -- All three panelists independently verified SI units for L_plasma [H], F_rad [N], P_brem [W/m^3], and rho_fill [kg/m^3].
- [x] **3+ peer-reviewed citations** -- Lee & Saw, J. Fusion Energy 27:292 (2008); Lee & Saw, J. Fusion Energy 33:319 (2014); Scholz et al., Nukleonika 51(1):79 (2006); Bosch & Hale, Nucl. Fusion 32:611 (1992); Miyoshi & Kusano, JCP 208:315 (2005); Borges et al., JCP 227:3191 (2008); Rybicki & Lightman, Radiative Processes (1979).
- [x] **Experimental evidence cited** -- Scholz (2006) PF-1000 I(t) waveform, Lee & Saw (2008) NX2 characterization.
- [x] **All assumptions explicitly listed** -- Each panelist provided numbered assumption lists with regimes of validity.
- [~] **Uncertainty budget** -- Discussed qualitatively (5% Rogowski, 8% NX2, Type B estimates). No formal GUM-compliant propagation through the calibration was performed. PARTIAL.
- [x] **All cross-examination criticisms addressed** -- Every Phase 2 challenge received a Phase 3 response or explicit concession.
- [x] **No unresolved logical fallacies** -- Chi-squared withdrawn (Dr. EE), "overfitting" retracted to "insufficient DOF", Vikhrev 1980 retracted (Dr. DPF).
- [x] **Explicit agreement/dissent** -- Dr. PP: 6.5 (AGREE with majority); Dr. DPF: 5.8 (DISSENT, lower); Dr. EE: 6.2 (AGREE with majority).

**Checklist result**: 8/9 pass, 1 partial (uncertainty budget).

---

## Supporting Evidence

### What Improved Since Debate #18

1. **CrossValidator pcf propagation bug FIXED**: `CrossValidator.validate()` now accepts and passes `pinch_column_fraction` to both `LeeModelCalibrator` and prediction `LeeModel`. 48 tests pass. (`calibration.py` lines 388-424)
2. **Phase AE test suite**: 22 tests covering pcf recalibration, cross-device prediction, published benchmarking, pcf parameter sweep, and NX2 characterization.
3. **Multiple incorrect claims cleaned up**: 9 retractions/concessions across the three panelists (see Concession Ledger below).

### What Was Confirmed As Deficient

1. **D2 molecular vs atomic mass density error**: `rho0 = n_fill * m_d` uses atomic deuterium mass (3.34e-27 kg) for molecular D2 gas. Factor-of-2 low density, absorbed by fm calibration. Present in both `lee_model_comparison.py:267` and `engine_validation.py:131`. NOT FIXED.
2. **21% timing error is model-form**: Present in both RLCSolver (implicit midpoint) and LeeModel (RK45). Root cause: snowplow lacks proper insulator flashover phase; liftoff delay (0.7 us) is a crude patch accounting for ~50% of the error.
3. **No post-D1 calibration run documented**: The single highest-value action -- running the calibrator after the D1 fix to verify fm in [0.05, 0.15] -- has not been performed. This was the P0 action from Debate #8.
4. **76% vs 33% current dip**: Partially addressed by pcf=0.14 (reduces to ~33% in engine_validation), but the Lee model comparison module still lacks reflected shock (Phase 4) physics.
5. **Zero MHD-vs-experiment comparisons**: All validation is 0D snowplow+circuit. The 2D/3D MHD engine has never been compared against experimental data.
6. **NX2 voltage mismatch**: Preset V0=14 kV vs experimental 400 kA at 11.5 kV (Sahyouni 2021). Creates 22% peak current scaling discrepancy.

---

## Scoring Decomposition

### Dr. PP (Pulsed Power) -- 6.5/10

| Subsystem | Score | Weight | Contribution |
|-----------|-------|--------|-------------|
| MHD Numerics | 8.0 | 0.18 | 1.44 |
| Transport | 7.5 | 0.12 | 0.90 |
| Circuit | 6.5 | 0.12 | 0.78 |
| DPF-Specific | 5.5 | 0.22 | 1.21 |
| Validation | 4.5 | 0.18 | 0.81 |
| AI/ML | 4.0 | 0.08 | 0.32 |
| Software | 7.5 | 0.10 | 0.75 |
| **Total** | | | **6.21 -> 6.5** |

Key quote: "The code is structurally sound but empirically unvalidated against experiment at the MHD level."

### Dr. DPF (Dense Plasma Focus) -- 5.8/10 (DISSENT)

| Component | Score | Weight | Contribution |
|-----------|-------|--------|-------------|
| Circuit Coupling | 6.0 | 0.25 | 1.50 |
| Snowplow Dynamics | 7.0 | 0.20 | 1.40 |
| Bremsstrahlung | 8.0 | 0.15 | 1.20 |
| Device Presets | 7.0 | 0.15 | 1.05 |
| Neutron Yield | 5.0 | 0.10 | 0.50 |
| Validation | 1.0 | 0.10 | 0.10 |
| Ablation | 0.0 | 0.05 | 0.00 |
| **Total** | | | **5.75 -> 5.8** |

Key quote: "I will not endorse any score above 6.0 until at least one published experimental I(t) waveform is quantitatively matched."

### Dr. EE (Electrical Engineering) -- 6.2/10

| Component | Score | Weight | Contribution |
|-----------|-------|--------|-------------|
| RLC Circuit Solver | 7.5 | 0.20 | 1.50 |
| Snowplow/Lee Physics | 5.5 | 0.20 | 1.10 |
| Validation Infrastructure | 6.0 | 0.15 | 0.90 |
| Calibration Framework | 5.0 | 0.15 | 0.75 |
| Uncertainty Quantification | 4.0 | 0.15 | 0.60 |
| Cross-Device Generalization | 4.5 | 0.10 | 0.45 |
| Documentation | 6.0 | 0.05 | 0.30 |
| **Total** | | | **5.60 + 0.6 D1/D2 credit = 6.2** |

Key quote: "The critical validation gap -- running the calibrator post-D1-fix and documenting the results -- remains unfilled."

### Moderator Assessment -- 6.3/10

The moderator score (6.3) is the weighted average of the three panelists (6.17) with a +0.1 uplift for the CrossValidator fix, which is a genuine infrastructure improvement delivered during this debate. The drop from 6.6 (Debate #18) reflects:
- Debate #18's 6.6 was slightly generous given that f_mr had zero physics impact (-0.1)
- D2 molecular mass error confirmed but not fixed (-0.1)
- No post-D1 calibration run (-0.1)
- Overall re-assessment is less favorable than previously assumed

---

## Assumptions and Limitations

1. **pcf=0.14 is a fit parameter**: Derived from Scholz X-ray imaging (z_f ~ 84 mm / 600 mm anode), but the analysis path is not published in a peer-reviewed paper. The value should be treated as a Type B estimate with +/-0.03 uncertainty.
2. **Lee model is Phase 1-2 only**: The `lee_model_comparison.py` module implements axial rundown and radial inward shock only. Reflected shock, slow compression, and expanded column phases are absent. The `snowplow.py` module does implement reflected shock (Phase 4).
3. **D2 molecular mass error**: Factor-of-2 low fill gas density throughout the validation code. Partially absorbed by fm calibration. Net physics impact is MEDIUM (2x-4x in density-dependent diagnostics).
4. **Liftoff delay uncertainty**: 0.7 us for PF-1000 with no stated uncertainty. Literature range: 0.5-1.5 us. This alone accounts for ~12% of the quarter-period, comparable to the timing error budget.
5. **Type B experimental uncertainties**: All experimental uncertainty values (5% PF-1000, 8% NX2) are expert estimates without published calibration certificates.

---

## Panel Positions

- **Dr. PP (Pulsed Power)**: **AGREE** with majority at 6.5/10. Infrastructure is strong, validation gap is the blocker. Path to 7.0 requires MHD-vs-experiment, reflected shock Phase 4, and blind cross-device prediction.
- **Dr. DPF (Dense Plasma Focus)**: **DISSENT** at 5.8/10. DPF physics fidelity is the correct axis to score on, not software quality. Without any experimental I(t) comparison, any score above 6.0 is premature. Milestones for improvement are well-defined.
- **Dr. EE (Electrical Engineering)**: **AGREE** with majority at 6.2/10. The post-D1 calibration run is the single highest-value action. The near-degenerate DOF problem limits calibration credibility.

---

## Dissenting Opinion (Dr. DPF -- 5.8/10)

Dr. DPF's dissent is grounded in the observation that the project scores highly on MHD numerics (8.0-8.5/10) and software engineering (7.5/10) but poorly on DPF-specific physics (5.5-6.0/10) and validation (1.0/10 from his perspective). When weighted by DPF physics importance rather than overall software quality, the composite drops to 5.8.

**Supporting argument**: "The simulator is now in a state where the next step is not more bug fixes but experimental validation. The code infrastructure supports running a full PF-1000 discharge. The question is no longer 'is the code correct?' but 'does the code match experiment?' A score of 5.8/10 appropriately captures this state: functioning but unvalidated."

**Moderator assessment of dissent**: The dissent is well-argued and reflects a legitimate weighting choice. For a DPF simulator, DPF physics fidelity should be weighted more heavily than MHD numerics quality. However, the MHD infrastructure enables the path to validation, and its quality deserves partial credit. The moderator's 6.3 reflects this balance.

---

## Concession Ledger (19 Total)

### Dr. PP (6 concessions given, 5 received)
**Given:**
1. CrossValidator bug claim retracted (code propagates pcf correctly post-fix)
2. pcf +/- 0.03 retracted (not GUM-compliant)
3. "Underdetermined" revised to "near-degenerate"
4. 55% dip estimate had unstated assumptions
5. ESR/ESL concern acknowledged as standard DPF practice (lumped L0)
6. Switch arc resistance, insulation coordination, electrode erosion explicitly retracted as scoring penalties

**Received:**
1. Dr. DPF: CrossValidator bug stale analysis
2. Dr. DPF: Vikhrev 1980 retracted
3. Dr. DPF: Circuit revised 7.2 -> 6.8
4. Dr. EE: Chi-squared withdrawn
5. Dr. EE: "Overfitting" retracted to "insufficient DOF"

### Dr. DPF (6 concessions given, 10 received)
**Given:**
1. No d(I_peak)/d(pcf) sensitivity computed
2. z_f is constant in Lee Phase 3 (initially challenged)
3. CrossValidator bug retracted (stale analysis)
4. Vikhrev 1980 citation retracted
5. Circuit score revised 7.2 -> 6.8
6. Liftoff delay accounts for ~50% of timing error

**Received:** (see Dr. PP and Dr. EE concessions above)

### Dr. EE (4 concessions given, 4 received)
**Given:**
1. Chi-squared withdrawn entirely
2. "Overfitting" retracted to "insufficient DOF"
3. Dip is envelope departure (conceded)
4. Peak error was compensating-error correction (conceded)

**Received:**
1. Dr. PP: CrossValidator bug claim retracted
2. Dr. DPF: No sensitivity analysis computed
3. Dr. DPF: Liftoff ~50% of timing error
4. 8% NX2 uncertainty is Type B (accepted by all)

---

## Recommendations for Further Investigation

### P0: Post-D1 Calibration Run (5 minutes, highest value)
Run `LeeModelCalibrator("PF-1000").calibrate()` and record optimized fc, fm, peak error, timing error, NRMSE. Verify fm in [0.05, 0.15]. This is the single action most likely to change the score.

### P1: Fix D2 Molecular Mass Error (30 minutes)
Replace `m_d` with `2*m_d` in fill density calculations in `lee_model_comparison.py:267` and `engine_validation.py:131`. Propagate to all density-dependent diagnostics.

### P2: Blind Cross-Device Prediction (1-2 hours)
Calibrate on PF-1000, predict NX2 I(t). If peak error < 10%, the model has genuine predictive power. Infrastructure exists in `CrossValidator` -- just run it with post-D1 parameters.

### P3: Reflected Shock Phase 4 in Lee Model (4-8 hours)
The `snowplow.py` has reflected shock; `lee_model_comparison.py` does not. Adding it would fix the 76% vs 33% dip discrepancy in the Lee model calibrator.

### P4: MHD Engine vs PF-1000 Experimental I(t) (8-16 hours)
The critical gap: no 2D/3D MHD simulation has been compared against experiment. Run the Metal engine for PF-1000 full discharge, compare against Scholz (2006). This would break the 7.0 ceiling.

### P5: NX2 Voltage Mismatch (30 minutes)
Update NX2 preset V0 from 14 kV to 11.5 kV (or add a second NX2 preset for the 11.5 kV operating point). Verify peak current scales correctly.

---

## Score Progression Update

| Debate | Score | Delta | Key Driver |
|--------|-------|-------|-----------|
| #18 | 6.6 | -0.3 | f_mr zero impact, 21% timing model-form, chi-squared withdrawn |
| **#19** | **6.3** | **-0.3** | **D2 density error confirmed, no post-D1 calibration, cleaned-up assessment less favorable** |

---

*PhD Debate #19 concluded. 5 phases executed. 19 concessions documented. 8/9 consensus checklist items passed.*
