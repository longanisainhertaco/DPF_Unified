# PhD Debate #40 — Phase BD: 3-Parameter Liftoff Delay Calibration

**Date**: 2026-03-01
**Question**: What is the current PhD-level academic assessment of the DPF-Unified simulator, considering Phase BD: 3-parameter liftoff delay calibration that reduces PF-1000 NRMSE from 14.3% to 9.6% and ASME ratio from 2.10 to 1.40?
**Previous Score**: Debate #39: 6.5/10 (CONSENSUS 3-0)

---

## VERDICT: CONSENSUS (3-0) — 6.5/10 (UNCHANGED)

### Answer

Phase BD implements a physically motivated 3-parameter calibration (`calibrate_with_liftoff()`) that jointly optimizes (fc, fm, liftoff_delay) using differential evolution. The NRMSE reduction from 14.3% to 9.6% is real as a calibration residual. However, the improvement cannot be cleanly attributed to the liftoff delay due to a critical fc bound asymmetry confound: the 3-parameter calibration uses fc_bounds=(0.5, 0.95) while the standard 2-parameter comparison is capped at fc <= 0.80. The resulting fc=0.9315 is outside the published range (0.6-0.8), ASME V&V 20 remains FAIL (ratio 1.40), and the definitive test (constrained-fc calibration) has not been run.

**Score: 6.5/10 — no change from Debate #39.**

---

## Panel Positions

| Panelist | Phase 1 | Phase 3 | Phase 4 (Final) | Verdict |
|----------|---------|---------|-----------------|---------|
| Dr. PP (Pulsed Power) | 6.5 | 6.5 | **6.5** | AGREE |
| Dr. DPF (Plasma Physics) | 6.6 | 6.35 | **6.45** | AGREE (rounds to 6.5) |
| Dr. EE (Electrical Engineering) | 6.5 | 6.5 | **6.5** | AGREE |

**Consensus range**: 6.45-6.5 (spread: 0.05 — tightest in Debate #40 series)

---

## Unanimous Agreement Points (7 items, 3-0, HIGH confidence)

### A1. T/4 for PF-1000 is ~10.5 us
- T = 2pi*sqrt(LC) = 2pi*sqrt(1.332e-3 * 33.5e-9) = 41.97 us
- **T/4 = 10.49 us** (not Dr. PP's erroneous 0.66 us, not Dr. DPF's T/2 = 20.99 us)
- Liftoff delay / T/4 = 0.705 / 10.49 = 6.7%
- **Implication**: Time-shift approximation IS valid for PF-1000

### A2. fc = 0.9315 is outside the published range
- Lee & Saw (2014): fc in [0.6, 0.8] across dozens of devices
- Mather (1971): fc typically 0.7 +/- 0.1
- Akel et al. (2016) citation **RETRACTED** by Dr. DPF (Lee model fit, not experimental measurement)
- No published experimental evidence for fc > 0.85 in any MJ-class Mather-type DPF

### A3. fc bound asymmetry confounds the NRMSE comparison
- `calibrate_with_liftoff()` uses fc_bounds = (0.5, 0.95) [line 687 of calibration.py]
- Standard 2-param calibration capped at fc <= 0.80 [line 785]
- The 33.8% NRMSE improvement conflates two effects: (a) wider fc bounds, (b) liftoff delay
- **Cannot attribute improvement to delay alone without constrained-fc experiment**

### A4. No direct PF-1000 leakage current measurement exists in cited literature
- Dr. PP: Gribkov et al. (2007) citation **RETRACTED** (paper addresses particle emission, not Rogowski leakage)
- Dr. EE: Scholz (2006) Fig. 5 attribution **RETRACTED** (paper shows total I(t) only)
- Dr. DPF: Akel et al. (2016) citation **RETRACTED** (unverifiable, Lee model fit)
- The fc = 0.6-0.8 range rests on Lee & Saw (2014) parametric fitting — effective parameter, not direct measurement

### A5. Liftoff delay is physically real
- Lee (2005): insulator flashover delay 0.5-1.5 us for MJ-class devices
- Saw (2011), PIC simulations confirm the phenomenon
- 0.705 us for PF-1000 is within the published range
- POSEIDON delay = 0 is consistent with device-specific physics (smaller device, faster flashover)
- **Implementation as pure time-shift is valid at 6.7% of T/4**

### A6. ASME V&V 20 ratio remains FAIL
- Ratio = E/u_val = 0.0955/0.0680 = 1.403 (need < 1.0 for PASS)
- With corrected u_val (including delay uncertainty): ratio ~ 1.37 (still FAIL)
- Improved from 2.121, but standard is binary: PASS or FAIL

### A7. The definitive test is constrained-fc liftoff calibration
- Run `calibrate_with_liftoff(fc_bounds=(0.6, 0.80))` to isolate delay contribution
- All three panelists agree this is the single most informative action
- Predicted outcomes:
  - If NRMSE drops from ~0.14 to ~0.11-0.12: delay genuinely useful (+0.1)
  - If NRMSE barely changes: improvement attributable to expanded fc bounds only (no credit)
  - If fc boundary-trapped at 0.80 with nonzero delay: partial credit (+0.05)

---

## Error and Retraction Summary

### Arithmetic/Citation Errors (3)
| Error | Panelist | Severity | Impact |
|-------|----------|----------|--------|
| T/4 = 0.66 us (should be 10.49 us) | Dr. PP | CRITICAL | Inverted A1 conclusion |
| T/4 = 20.99 us (should be 10.49 us) | Dr. DPF | MODERATE | Used T/2 formula, correct qualitative conclusion |
| tau_flash volume vs surface mechanism | Dr. DPF | MODERATE | Wrong breakdown physics |

### Full Retractions (6)
| Claim | Panelist | Reason |
|-------|----------|--------|
| t_liftoff/T/4 = 1.07 | Dr. PP | Unit error (uF vs mF) |
| Gribkov et al. (2007) Rogowski leakage | Dr. PP | Wrong paper for claimed content |
| Scholz (2006) Fig. 5 leakage decomposition | Dr. EE | Paper shows total I(t) only |
| Oberkampf & Roy p/N_eff < 0.25 threshold | Dr. EE | Could not verify citation |
| Akel et al. (2016) fc range extension | Dr. DPF | Unverifiable, Lee model fit not measurement |
| tau_flash formula (volume breakdown) | Dr. DPF | Should be surface flashover (Boersch-Toepfer) |

### Concessions (18 total)
| Panelist | Full Concessions | Partial Concessions | Notes |
|----------|-----------------|---------------------|-------|
| Dr. PP | 5 | 0 | A1 unit error, leakage channels, compensating unfalsifiable, ASME closability, universal liftoff |
| Dr. DPF | 10 | 0 | Most concessions in panel; 2 full retractions; revised score from 6.6 to 6.35 to 6.45 |
| Dr. EE | 4 | 2 | u_val estimate, Oberkampf threshold, "any monotonic signal", Scholz Fig. 5 |

---

## Remaining Disagreements (3)

### D1. Exact score magnitude (6.35-6.5)
- Dr. DPF revised down from 6.6 to 6.35 (Phase 3) then up to 6.45 (Phase 4)
- Dr. PP and Dr. EE both hold 6.5
- **Resolution**: 6.5 — Phase BD is neutral (adds infrastructure but doesn't advance physics or validation)

### D2. DPF-Specific credit for liftoff delay
- Dr. PP: 0 (pure time-shift, not physics)
- Dr. DPF: 0 (retracted +0.1 after Phase 3 concessions)
- Dr. EE: +0.05 max (physically motivated, rounds to 0 at composite level)
- **Resolution**: 0 at composite level. Credit conditional on constrained-fc test.

### D3. fc^2/fm = 8.05 interpretation
- All agree it departed from the 4.7-6.8 degeneracy ridge
- Likely artifact of expanded fc bounds, not physics insight
- **Resolution**: Requires constrained-fc experiment to settle

---

## Sub-Scores

| Category | Score | Change | Justification |
|----------|-------|--------|---------------|
| MHD Numerics | 8.0 | 0.0 | No changes |
| Transport Physics | 7.0-7.5 | 0.0 | No changes |
| Circuit Model | 6.8 | 0.0 | RLC solver unchanged; delay is calibration layer |
| DPF-Specific | 6.0-6.2 | 0.0 | Time-shift is not physics; conditional +0.05 pending test |
| V&V | 5.5 | 0.0 | ASME FAIL; confounded comparison; Section 5.1 violation |
| AI/ML | 4.5 | 0.0 | No changes |
| Software | 7.6 | 0.0 | `calibrate_with_liftoff()` well-implemented |

---

## Consensus Verification Checklist

- [x] **Mathematical derivation provided**: T/4 calculation verified by all panelists (with corrections)
- [x] **Dimensional analysis verified**: SI units checked on T/4, fc^2/fm, u_val
- [x] **3+ peer-reviewed citations**: Lee & Saw (2014), Lee (2005), Mather (1971), Saw (2011), Burnham & Anderson (2002), Oberkampf & Trucano (2002)
- [x] **Experimental evidence cited**: Lee (2005) flashover delay measurements, Lee & Saw (2014) fc range survey
- [x] **All assumptions explicitly listed**: fc range, N_eff ~5, u_val components, T/4 timing regime
- [x] **Uncertainty budget**: u_val = 0.0680 (without delay), ~0.070 (with delay uncertainty, quadratic estimate)
- [x] **All cross-examination criticisms addressed**: 18 concessions, 6 retractions, all criticisms responded to in Phase 3
- [x] **No unresolved logical fallacies**: "Universal liftoff" strawman conceded; "compensating errors" noted as unfalsifiable
- [x] **Explicit agreement from each panelist**: Dr. PP AGREE (6.5), Dr. DPF AGREE (6.45, rounds to 6.5), Dr. EE AGREE (6.5)

---

## Path to 7.0 (Revised)

| Action | Expected Delta | Priority | Effort |
|--------|---------------|----------|--------|
| Constrained-fc test: fc <= 0.80 with free delay | +0.05-0.10 | **HIGHEST** | 5 min compute |
| Wire MC with delay into ASME u_val | +0.0 (fixes bug) | HIGH | 5 LOC |
| Third device with digitized I(t) | +0.1-0.2 | HIGH | Research + digitize |
| Physical liftoff model (unloaded circuit during delay) | +0.05-0.10 | MEDIUM | ~200 LOC |
| Combine delay with circuit_only_calibration() for Section 5.3 | +0.05 | MEDIUM | ~50 LOC |
| AIC/BIC model selection: 2-param vs 3-param (same bounds) | +0.05 | LOW | ~30 LOC |
| ASME PASS (ratio < 1.0) | +0.2 | VERY HIGH | May require physics improvement |

**If constrained-fc + third device + MC wiring completed: potential +0.2-0.4, reaching 6.7-6.9.**

**7.0 ceiling NOT broken (40th consecutive debate).**

---

## Key Insight

Phase BD's most valuable contribution is *diagnostic, not predictive*: it revealed the fc bound asymmetry confound (line 687 vs 785 of calibration.py), exposed the absence of direct PF-1000 leakage measurements in the literature, and provided a framework for 3-parameter optimization. The infrastructure is sound; the headline NRMSE result is confounded. The constrained-fc experiment is the single most informative next step.

---

## Score Progression (Debates #38-40)

| Debate | Score | Delta | Key Driver |
|--------|-------|-------|-----------|
| #38 | 6.5 | 0.0 | Phase BB.3 fixes, Bennett bugs found+fixed |
| #39 | 6.5 | 0.0 | Phase BC circuit-only calibration |
| **#40** | **6.5** | **0.0** | **Phase BD liftoff delay — fc bound confound, fc=0.93 non-physical, ASME FAIL** |
