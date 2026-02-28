# PhD Debate #26 Verdict: Phase AN Blind NX2 Cross-Device Prediction

## VERDICT: MAJORITY (2-1) — 6.4/10 (-0.1)

### Question
Does Phase AN (Blind NX2 cross-device prediction using PF-1000-calibrated fc=0.816, fm=0.142) break the 7.0 ceiling?

### Answer
**NO.** Phase AN does not break the 7.0 ceiling. Score drops 0.1 to 6.4/10.

Phase AN was the panel's highest-priority recommendation from Debate #25: blind cross-device prediction without re-fitting. The methodology is sound (degradation factor, sensitivity analysis, 12 tests all passing), but the results reveal two fatal problems: (1) the NX2 experimental reference data has irreconcilable parameter ambiguities (R0, fill gas species, fill pressure), and (2) the fc²/fm degeneracy makes the blind-vs-native comparison insensitive to the transferred parameters.

### Phase AN Key Results

| Metric | Blind (PF-1000 fc/fm) | Native (NX2 fc/fm) | Experiment |
|--------|----------------------|--------------------|-----------|
| Peak current | 280.5 kA | 281.0 kA | 400 kA ±8% |
| Peak timing | 1.02 µs | 1.05 µs | 1.8 µs ±12% |
| Peak error | 29.9% | 29.7% | — |
| Timing error | 43.3% | 41.9% | — |
| Degradation factor | 1.0x | (baseline) | — |

### Consensus Verification Checklist

- [x] Mathematical derivation provided — RLC peak current, snowplow dynamics derived
- [x] Dimensional analysis verified — all three panelists checked SI units
- [x] 3+ peer-reviewed citations — Sahyouni (2021, DOI:10.1155/2021/6611925), Lee & Saw (2008), ASME V&V 20-2009, Khan (2014, DOI:10.1155/2014/240729)
- [x] Experimental evidence cited — NX2 400 kA from Lee & Saw (2008) Table 1
- [x] Assumptions explicitly listed — 8 assumptions identified (A1-A8), 3 suspect
- [x] Uncertainty budget — GUM budget computed (u_val=19.0%), ASME E/u_val=1.57 (FAIL)
- [x] All cross-examination criticisms addressed — 10 resolved, 4 remaining
- [x] No unresolved logical fallacies — 2-sigma fallback retracted, R0 contamination retracted
- [x] Explicit agreement/dissent — Dr. PP 6.5 (dissent), Dr. DPF 6.4, Dr. EE 6.4

### Supporting Evidence

**1. The 1.0x degradation is a Lee model degeneracy, not transfer evidence.**

The Lee model's axial dynamics depend on fc²/fm, not fc and fm independently:
- PF-1000 calibrated: fc²/fm = 0.816²/0.142 = 4.69
- NX2 native: fc²/fm = 0.700²/0.100 = 4.90

These ratios differ by only 4.5%, explaining why blind and native give nearly identical results. The sensitivity analysis confirms: dI/dfc = -59 kA/unit and dI/dfm = +166 kA/unit produce compensating effects (Δ = -6.8 + 7.0 = +0.2 kA net). This degeneracy was documented by Lee himself and is a structural property of the model, not evidence of universal fc/fm.

All three panelists agree on this point after Phase 2 cross-examination.

**2. The 30% peak current offset is dominated by NX2 parameter ambiguity.**

| Parameter | Code Value | RADPF Model Page | Discrepancy |
|-----------|-----------|-----------------|-------------|
| R0 | 5 mΩ | 2.3 mΩ | 2.2x |
| Fill gas | D2 at 4 Torr | Neon at 2.63 Torr | Different species |
| fm | 0.1 (native) | 0.0635 | 1.6x |
| L0 | 20 nH | 15 nH (Sahyouni) | 1.3x |

With R0 = 5 mΩ, the unloaded RLC peak is 374.5 kA — already 6.4% below the experimental 400 kA before any plasma loading. The R0 = 2.3 mΩ value cited by Dr. PP was retracted in Phase 3 as PF-1000 contamination (Scholz 2006, not NX2).

**3. ASME V&V 20 validation formally fails.**

GUM budget (approximate, independence assumed):
- u_exp = 8.5% (NX2 measurement, Type B)
- u_num = 7.4% (MHD spatial discretization, not BDF2 temporal)
- u_input = 15.3% (R0, L0, gas species, fill pressure)
- Combined: u_val = 19.0%
- E/u_val = 29.9%/19.0% = 1.57 → **FORMAL FAIL**

The 2-sigma "marginal pass" language was retracted by Dr. EE in Phase 3.

**4. NX2 fill pressure inconsistency is a systematic error, not an uncertainty.**

The NX2 preset uses 133 Pa (1 Torr) while experimental.py lists 4 Torr (532 Pa). This factor-of-4 discrepancy in fill density produces ~2x timing error by the snowplow scaling t_pinch ∝ √ρ₀. All three panelists agree this must be corrected before any GUM budget is meaningful.

### Phase 3 Concessions (10 resolved points)

1. R0 = 2.3 mΩ is PF-1000, not NX2 (Dr. PP conceded)
2. L_plasma decomposition (7.7 + 15.37 = 23.05 nH) arithmetically wrong (Dr. DPF conceded)
3. L_total = 47 nH and I_peak = 400 kA cannot be simultaneous (Dr. DPF conceded)
4. 2-sigma "marginal pass" not in ASME V&V 20 (Dr. EE conceded)
5. fc²/fm degeneracy is known Lee model property, not coincidence (Dr. PP conceded)
6. fc/fm covariance nonzero, needed in GUM budget (Dr. EE conceded)
7. Fill pressure inconsistency is systematic error, not GUM uncertainty (Dr. EE conceded)
8. Species mass scaling √2 required for H2→D2 comparisons (Dr. EE conceded)
9. u_num label should specify MHD spatial discretization (Dr. EE clarified)
10. Parasitic inductance budget double-counts Scholz L0 (Dr. PP conceded)

### Panel Positions

- **Dr. PP (Pulsed Power): 6.5/10 — DISSENT (+0.1)**
  - Credits Phase AN for honest methodology: degradation factor, sensitivity analysis, 12 passing tests
  - "The 30% error is from uncertain NX2 reference data, not from model regression"
  - Concedes R0 cross-device contamination, fc²/fm degeneracy, parasitic double-counting

- **Dr. DPF (Plasma Physics): 6.4/10 — AGREE**
  - "Phase AN reveals the fc²/fm degeneracy makes blind-vs-native insensitive — the test has no discriminating power"
  - "The 30% offset must be decomposed into systematic input error vs model form error before credit is given"
  - Concedes L_plasma arithmetic errors, L_total energy contradiction, fm citation gap

- **Dr. EE (Electrical Engineering): 6.4/10 — AGREE**
  - "ASME V&V 20 formal FAIL at E/u_val = 1.57. No 2-sigma fallback."
  - "NX2 experimental characterization is dramatically worse than PF-1000 — no waveform, ambiguous parameters"
  - Concedes GUM independence assumption, species scaling, fill pressure systematic error

### Dissenting Opinion (Dr. PP)

The score should remain at 6.5/10 because Phase AN added real scientific content:
1. 12 new tests (9 non-slow, 3 slow), all passing, expanding test coverage to 2868 non-slow
2. The degradation factor methodology is correct even if the result is explained by degeneracy
3. The sensitivity analysis (dI/dfc, dI/dfm) provides quantitative Jacobian elements for future uncertainty propagation
4. Discovering the NX2 parameter ambiguity (R0, gas species, fill pressure) is itself a valuable finding

The 0.1 drop penalizes the project for performing an honest test that revealed model limitations. The code is not worse than before Phase AN — it is better characterized.

### Sub-Scores (Updated)

| Category | Debate #25 | Debate #26 | Change | Justification |
|----------|-----------|-----------|--------|---------------|
| Circuit model | 6.8 | 6.8 | 0 | NX2 circuit runs correctly; R0 ambiguity is data quality, not code quality |
| MHD physics | 6.0 | 6.0 | 0 | No MHD changes in Phase AN |
| Validation (V&V) | 5.5 | 5.3 | -0.2 | NX2 blind prediction FAILS ASME; reveals worse-than-expected experimental characterization |
| DPF-specific | 7.5 | 7.5 | 0 | Snowplow model unchanged |
| Software eng. | 7.0 | 7.0 | 0 | Test infrastructure sound |
| Numerical methods | 6.5 | 6.5 | 0 | No numerical changes |

### Recommendations for Further Investigation

**HIGHEST PRIORITY (to break 7.0):**
1. **Fix NX2 R0 and fill conditions**: Determine correct R0 for D2 operation from Lee & Saw (2008) Table 1 or RADPF model. Correct fill pressure in presets.py.
2. **Re-run Phase AN with corrected parameters**: If R0 correction brings native NX2 within 10% of 400 kA, the blind prediction becomes meaningful.
3. **Obtain NX2 digitized I(t) waveform**: The RADPF code outputs I(t) waveforms that could serve as a synthetic experimental reference.

**HIGH PRIORITY:**
4. **Add a third device**: UNU-ICTP (best-documented small DPF) would provide a genuine three-device cross-validation.
5. **Compute Hessian at calibration optimum**: This gives the fc-fm covariance matrix needed for a GUM-compliant uncertainty budget.
6. **Propagate fc²/fm through the model**: Since fc²/fm is the invariant, calibrate fc²/fm directly rather than fc and fm separately.

### Score Progression

| Debate | Score | Change | Key Phase |
|--------|-------|--------|-----------|
| #20 | 6.3 | — | Baseline |
| #21-24 | 6.3 | 0 | Verification loop (AH-AL) |
| #25 | 6.5 | +0.2 | Phase AM: Metal I(t) validation |
| **#26** | **6.4** | **-0.1** | **Phase AN: NX2 blind prediction fails** |

### 7.0 Ceiling Status: NOT BROKEN (26th consecutive debate)
