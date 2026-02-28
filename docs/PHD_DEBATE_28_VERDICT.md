# PhD Debate #28 Verdict: Phase AP Timing-Based Validation

## VERDICT: CONSENSUS (3-0) — 6.5/10 (HOLD, unchanged from Debate #27)

### Question
Does Phase AP (timing-based cross-device validation + data integrity fixes, 29 tests) justify raising the score above 6.5/10?

### Answer
**No.** Phase AP competently implements the three Debate #27 recommendations (timing metrics, UNU-ICTP waveform, NX2 L0 sweep) but the results expose new problems rather than demonstrating improved validation. NX2 timing error of 52.5% is a model failure, the ASME V&V 20 framework has a structural error (u_model in u_val), and the headline NRMSE=0.034 measures model self-consistency rather than predictive power.

### Panel Positions
- **Dr. PP (Pulsed Power): AGREE 6.5** — Timing metric exposes deeper NX2 failure. First-peak finder bug confirmed (returns axial transit time, not true peak). ASME V&V 20 produces FAIL on first attempt.
- **Dr. DPF (Plasma Physics): AGREE 6.5** — NRMSE=0.034 is structural consequence of L_p << L0. Cube-root suppression limits timing discrimination to 6.2% (smaller than 18.4% model-form error). No new physics implemented.
- **Dr. EE (Electrical Engineering): AGREE 6.5** — ASME V&V 20 u_val improperly includes u_model per Section 2.4. All uncertainties Type B without calibration backing. Effective validation remains single-device (PF-1000 only).

### Supporting Evidence

#### Consensus Verification Checklist
- [x] Mathematical derivation provided — cube-root suppression from snowplow equation
- [x] Dimensional analysis verified — L_p/L0 ratio, transit time scaling
- [x] 3+ peer-reviewed citations — Lee & Saw (2008, 2009, 2014), Scholz (2006), ASME V&V 20-2009, GUM JCGM 100:2008
- [x] Experimental evidence cited — PF-1000 Scholz waveform (only device with digitized data)
- [x] All assumptions explicitly listed — 10 assumptions with regime of validity
- [x] Uncertainty budget — Type B estimates acknowledged; u_model/u_val structural error identified
- [x] All cross-examination criticisms addressed — 5 contested points resolved
- [x] No unresolved logical fallacies — f_mr confound resolved by evidence
- [x] Explicit agreement from each panelist — 3-0 CONSENSUS

### Sub-Score Breakdown

| Category | Debate #27 | Phase AP | Delta | Rationale |
|----------|-----------|----------|-------|-----------|
| Physics Fidelity | 7.0 | 7.0 | 0 | No new physics models |
| Numerical Methods | 7.2 | 7.2 | 0 | No algorithm changes |
| Software Engineering | 7.5 | 7.6 | +0.1 | 29 well-structured tests, data integrity fixes |
| Circuit Model | 6.8 | 6.8 | 0 | L0 sweep characterizes but doesn't improve |
| V&V Framework | 5.5 | 5.8 | +0.3 | Timing metrics, ASME V&V 20 attempted, NX2 loading analysis |
| Cross-Device Validation | 5.0 | 5.1 | +0.1 | 3x3 timing matrix; NX2 still broken (52.5%) |

### Key Findings

1. **First-peak finder bug (CONFIRMED)**: NX2 `_find_first_peak` returns axial-to-radial transition time (0.855 us) not true current peak (0.991 us). Inflates timing error from 44.9% to 52.5%. Fix: require sustained decline or use argmax for non-ringing waveforms.

2. **ASME V&V 20 construction error (CONFIRMED)**: u_model=10% included in u_val denominator. Per Section 2.4, model-form error is the OUTPUT of validation, not an INPUT. Removing u_model: ratio increases from 1.023 to 1.227 (clearer FAIL).

3. **NRMSE=0.034 is self-consistency (CONFIRMED)**: Compares blind Lee model vs native Lee model with identical device parameters. Since L_p/L0 ~ 0.25-0.35 for UNU-ICTP, any two Lee model runs produce similar waveforms. Not a validation result.

4. **f_mr confound RESOLVED**: Fixed-f_mr and variable-f_mr give identical results (6.98% peak, 6.22% timing) because f_mr only affects post-peak radial phase.

5. **NX2 400 kA CONFIRMED model-derived**: At L0=20 nH, unloaded peak=402.2 kA with 0.5% loading (unphysical). At L0=15 nH, loading=35% (physical). The reference value is the unloaded RLC peak, not a measurement.

### Concessions (Phase 2+3)
- Dr. PP: Accepted +0.2 V&V credit (up from +0.1) after Dr. EE argued framework-that-reports-FAIL is valuable
- Dr. EE: Reduced V&V credit to +0.4 (down from +0.5) after u_model structural error acknowledged
- Dr. DPF: Accepted binomial expansion requires more care for UNU-ICTP (L_p/L0=0.35, not negligible)

### Recommendations (Actionable for Next Phase)

#### Immediate Fixes (Bug/Method)
1. Fix `_find_first_peak` for NX2 — use argmax or sustained-decline criterion
2. Remove u_model from u_val in ASME V&V 20 assessment
3. Assert on ASME V&V 20 result for PF-1000 (should PASS: 9.3% error, ~15% u_exp)

#### Score-Increasing Work
4. Obtain digitized UNU-ICTP or NX2 experimental waveform for model-vs-experiment comparison
5. Add liftoff delay model (~0.2-0.5 us) to reduce UNU-ICTP timing error from 18.4%
6. Resolve NX2 L0 to 15 nH (where physics is sensible) and re-run timing validation
7. Demonstrate framework can distinguish correct vs incorrect physics (not just "everything gets ~7% error")

### Score Progression

| Debate | Score | Delta | Phase |
|--------|-------|-------|-------|
| #25 | 6.5 | +0.2 | Phase AM Metal I(t) validation |
| #26 | 6.4 | -0.1 | Phase AN blind NX2 (fc^2/fm degeneracy) |
| #27 | 6.5 | +0.1 | Phase AO three-device cross-prediction |
| **#28** | **6.5** | **0.0** | **Phase AP timing validation (HOLD)** |

### 7.0 Ceiling Analysis (28th consecutive debate below 7.0)

The 7.0 ceiling persists because the project has **one validated device** (PF-1000) with **one digitized waveform** and **two fitted parameters** (fc, fm). All cross-device work compares model-vs-model, not model-vs-experiment. Breaking 7.0 requires:
- A second independent experimental waveform (not model-derived)
- Demonstrated prediction (not post-hoc fit) against that waveform
- ASME V&V 20 PASS for at least two devices

---
*PhD Debate #28, 2026-02-28. Moderator: Claude Opus 4.6*
*Panel: Dr. PP (Pulsed Power), Dr. DPF (Plasma Physics), Dr. EE (Electrical Engineering)*
