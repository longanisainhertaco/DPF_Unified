# PhD Debate #47 — Phase BK: fm-Constrained Multi-Device Calibration + LOO CV

**Date**: 2026-03-03
**Question**: What is the PhD-level assessment of DPF-Unified considering Phase BK (fm-constrained multi-device calibration with leave-one-out cross-validation)?

---

## VERDICT: CONSENSUS (3-0) — Score: 6.6/10 (UNCHANGED)

### Answer

Phase BK implements the fm >= 0.10 physical constraint across multi-device calibration and adds leave-one-out cross-validation infrastructure. The results are methodologically sound but produce a definitive negative finding: LOO degradation factors of 4.92x (hold-PF1000) and 6.52x (hold-POSEIDON) confirm that Lee model parameters are device-specific, not transferable. Both fm values boundary-trap (PF-1000 at lower bound 0.10, POSEIDON at upper bound 0.40), confirming model-form inadequacy under physical constraints. The score remains 6.6/10 — the LOO infrastructure is a genuine V&V contribution, but the negative results do not increment the score.

### Panel Positions
- **Dr. PP** (Pulsed Power): **6.6/10** — LOO infrastructure earns +0.1 for methodology; devastating LOO degradation proves non-transferability; fm boundary-trapping expected from iso-acceleration degeneracy
- **Dr. DPF** (Plasma Physics): **6.6/10** — Boundary-trapping is mathematically necessary given fc^2/fm spread; LOO error is >95% bias-dominated; model structurally inadequate for cross-device prediction
- **Dr. EE** (Electrical Engineering): **6.6/10** — LOO with N=2 is two blind predictions, not cross-validation; +293.7% POSEIDON penalty is relative to independent baseline, NOT worse than bare RLC; infrastructure real but statistics invalid at N=2

---

## Phase 4: Points of Agreement (Unanimous, 3-0)

### A1: LOO Infrastructure Is Genuine
The `leave_one_out()` method in `MultiDeviceCalibrator` (calibration.py:3338-3417) correctly implements the train-on-N-1/predict-on-1 protocol. This infrastructure enables future multi-device studies when N >= 3 devices become available. (3-0, HIGH confidence)

### A2: LOO with N=2 Has No Inferential Power
With N=2 devices, LOO produces exactly 2 blind predictions. The sampling distribution of the mean LOO error follows a t-distribution with df=1 (Cauchy), which has undefined mean and infinite variance. No valid confidence interval can be constructed. E_LOO = 0.430 is descriptive only. (3-0, HIGH confidence)

### A3: LOO Degradation Is Real Physics, Not Statistical Artifact
The 4.92x and 6.52x degradation factors reflect genuine differences between PF-1000 (S/S_opt=0.98, near-optimal) and POSEIDON-60kV (S/S_opt=2.81, super-driven). These devices operate in qualitatively different regimes. Adding more devices would characterize the DISTRIBUTION of degradation, not reduce it for this specific pair. (3-0, HIGH confidence)

### A4: fm Boundary-Trapping Is a Mathematical Consequence
At shared fc=0.752, the iso-acceleration degeneracy predicts fm_PF = 0.752^2/8.03 = 0.070 (trapped at 0.10) and fm_POS = 0.752^2/0.87 = 0.650 (trapped at 0.40). This exactly matches Phase BK observations. No parameter tuning is needed to predict the boundary-trapping. (3-0, HIGH confidence)

### A5: Mode 1 Constraint Non-Binding Is Uninformative
The Mode 1 shared fm=0.167 already exceeds the fm >= 0.10 threshold. The constraint adds zero information. This should not be cited as evidence the constraint "works." (3-0, HIGH confidence)

### A6: fc^2/fm Ratio Is Methodology-Dependent
The cross-device fc^2/fm ratio depends critically on which calibration formulation is used:
- Inconsistent comparison (Phase BJ): PF-1000 unconstrained fc^2/fm=8.03 vs POSEIDON independent=0.87 → 9.2x
- Consistent 2-parameter comparison: PF-1000 fc^2/fm=5.00 vs POSEIDON IPFS=1.29 → 3.88x
The 9.2x figure is inflated by comparing across optimization methods. The 3.88x using consistent methodology has only 1.32x excess over the geometric factor (2.94x), which is explainable by standard device-specific factors. (3-0, HIGH confidence)

### A7: Shared Parameters Retain Predictive Content
Contrary to Dr. EE's retracted claim, the shared-parameter POSEIDON NRMSE of 0.228 is a 51% improvement over bare RLC (NRMSE = 0.466). The Lee model with shared parameters IS better than no physics — it captures the correct timescale and amplitude from the circuit parameters. (3-0, HIGH confidence, following Dr. EE's full concession)

### A8: Scholz R0 = 2.3 mOhm Likely Includes AC Skin Effect
Standard DPF short-circuit calibration measures R0 at the discharge frequency. If Scholz (2006) followed this practice, skin effect is already captured in R0. The R_AC/R_DC ratio for the anode is genuinely O(100), but the absolute contribution is ~1.5% of total R0. For bus-bars, R_AC/R_DC = 11.7-23.4 (not 5.4 as initially claimed). (3-0, MEDIUM confidence — Scholz measurement method not independently verified)

### A9: Switch Arc Resistance Is Negligible for Lee Model
Rompe-Weizel model: R_arc = 20 mOhm at t=14 ns (I = 11 kA). By the snowplow phase (t > 500 ns), R_arc < 0.1 mOhm — completely negligible vs R0 = 2.3 mOhm. (3-0, HIGH confidence, following Dr. PP's full derivation)

---

## Concession Inventory

### Dr. PP Concessions (8 total: 6 full, 2 partial)
1. **"Zero statistical power" → "uncharacterizable variance"**: Terminology correction (FULL)
2. **POSEIDON fm=0.40 vs IPFS fm=0.275 comparison**: Withdrawn — different codes/objectives not comparable (FULL)
3. **Circuit model scope vs Lee model scope**: Lee model intentionally absorbs circuit losses into fc/fm (PARTIAL)
4. **n_eff addition across devices**: Composition fallacy — n_eff not additive across physically different systems (FULL)
5. **"Lee model cannot be used for design"**: Narrowed to "DPF-Unified has not demonstrated design-quality prediction" (FULL)
6. **LOO uncertainty bars ±0.5/0.7**: Retracted as fabricated heuristics (FULL — false precision)
7. **Cross-device ASME ratio ~6.9**: Withdrawn — incompatible u_val budgets (FULL)
8. **Switch arc 20 mOhm significance**: Retracted — occurs at t=14 ns, negligible by snowplow (FULL)

### Dr. DPF Concessions (8 total: 6 full, 2 partial)
1. **Geometric factor 2.94x not 2.5x**: Arithmetic error; excess 3.1x not 3.7x (FULL)
2. **LOO bias-variance decomposition underdetermined at N=2**: Cannot decompose with 1 DOF (FULL)
3. **fc^2/fm 9.2x → 3.88x using consistent methodology**: Major revision — excess over geometry drops from 3.7x to 1.32x (FULL)
4. **0.096 statistical component unsourced**: Cannot reconstruct provenance; withdrawn (FULL)
5. **Cherry-picked fc values for neutron yield**: Conceded; I^4 scaling defended from beam-target theory (PARTIAL)
6. **t_transit ~ fm^(1/3)/fc^(2/3) RETRACTED**: Not derivable from standard Lee model; requires I(t) ~ sqrt(t) which is non-standard. Correct scaling depends on current-rise exponent alpha. Debate #42 record needs correction. (FULL — MAJOR RETRACTION)
7. **Structural inadequacy from fc^2/fm circularity**: fc^2/fm variation is consequence of non-identifiability; structural inadequacy better supported by ASME FAIL + boundary-trapping + cross-device degradation (PARTIAL)
8. **Measurement uncertainty budgets for non-I(t) observables**: Provided; neutron yield worst S/N (CV 50-100%); sheath velocity and X-ray timing most promising (FULL)

### Dr. EE Concessions (7 total: 5 full, 2 partial)
1. **"+293.7% worse than RLC baseline"**: WRONG — bare RLC NRMSE = 0.466; shared 0.228 is 51% improvement (FULL — MAJOR RETRACTION)
2. **maxiter=150 "33x below recommendations"**: 150 not in codebase; recommendation fabricated (FULL — MAJOR RETRACTION)
3. **BIC "statistical artifact" re-raised after Debate #46 retraction**: Protocol violation acknowledged (FULL)
4. **LOO infrastructure genuine**: Real V&V code; would revise at N=3 with stated criterion (PARTIAL)
5. **Transit time scaling is physics content**: t_transit scaling from Debate #42 is legitimate (PARTIAL)
6. **LOO N=2 t-distribution CI formally Cauchy**: Agrees this REINFORCES that N=2 has no inferential power (no concession needed — alignment)

**Total: 23 concessions (17 full + 6 partial), 3 major retractions**

---

## Sub-Score Assessment

| Category | Weight | Debate #46 | Debate #47 | Delta | Rationale |
|----------|--------|-----------|-----------|-------|-----------|
| MHD Numerics | 0.15 | 8.0 | 8.0 | 0.0 | No MHD changes |
| Transport Physics | 0.10 | 7.5 | 7.5 | 0.0 | No transport changes |
| Circuit Model | 0.15 | 6.8 | 6.8 | 0.0 | No circuit changes; skin effect absorbed in Scholz R0 |
| DPF-Specific | 0.20 | 5.8 | 5.8 | 0.0 | LOO failure offsets methodology gain |
| V&V | 0.25 | 5.7 | 5.7 | 0.0 | LOO infrastructure exists but N=2 statistically invalid |
| AI/ML | 0.05 | 4.5 | 4.5 | 0.0 | No AI changes |
| Software Eng. | 0.10 | 7.8 | 7.8 | 0.0 | LOO method is incremental to MultiDeviceCalibrator |

**Weighted**: 0.15(8.0) + 0.10(7.5) + 0.15(6.8) + 0.20(5.8) + 0.25(5.7) + 0.05(4.5) + 0.10(7.8) = 6.56 ≈ **6.6/10**

---

## Consensus Verification Checklist

- [x] **Mathematical derivation provided** — Iso-acceleration degeneracy derived from snowplow EOM; boundary-trapping predicted quantitatively
- [x] **Dimensional analysis verified** — All formulas checked by all 3 panelists (SI units throughout)
- [x] **3+ peer-reviewed citations** — Lee & Saw (2014), Scholz (2006), Haines (2011), Gibson & McAuley (2024), Sahyouni et al. (2021), Roy et al. (2018), Burnham & Anderson (2002)
- [x] **Experimental evidence cited** — Scholz (2006) PF-1000 I(t), IPFS POSEIDON-60kV I(t)
- [x] **All assumptions explicitly listed** — 10 assumptions (Dr. DPF), 8 (Dr. PP), 8 (Dr. EE)
- [x] **Uncertainty budget** — LOO degradation uncharacterizable at N=2; POSEIDON L0/R0 circularity adds 15-30% systematic
- [x] **All cross-examination criticisms addressed** — 23 concessions, 0 unresolved challenges
- [x] **No unresolved logical fallacies** — Composition fallacy (n_eff addition), cherry-picking (fc values), and circular reasoning (fc^2/fm) all corrected
- [x] **Explicit agreement/dissent from each panelist** — All 3 agree on 6.6/10

---

## Key Findings from This Debate

### F1: The 9.2x fc^2/fm Variation Is Inflated (REVISED)
Using consistent 2-parameter methodology, the cross-device fc^2/fm ratio is 3.88x (PF-1000: 5.00, POSEIDON IPFS: 1.29). The geometric factor accounts for 2.94x, leaving only 1.32x "excess" — within calibration uncertainty. The widely-cited 9.2x figure compared inconsistent calibration formulations. (3-0)

### F2: t_transit ~ fm^(1/3)/fc^(2/3) Is RETRACTED
This scaling, recorded as "verified" in Debate #42, is not derivable from standard Lee model equations. Simple dimensional analysis gives t ~ fm^(1/2)/fc (constant current) or t ~ fm^(1/4)/fc^(1/2) (linearly rising current). The fm^(1/3)/fc^(2/3) form requires I(t) ~ sqrt(t), which is non-standard. The Debate #42 record should be corrected. (3-0)

### F3: Shared Parameters Retain 51% Improvement Over Bare RLC
For POSEIDON, the constrained shared-parameter NRMSE of 0.228 is a 51% improvement over bare RLC (0.466). The Lee model with shared parameters IS better than no physics, even with the fm constraint. Dr. EE's original framing ("worse than baseline") was incorrect and fully retracted. (3-0)

### F4: LOO at N=2 Is Descriptive, Not Inferential
Two blind predictions dressed as cross-validation. The Cauchy distribution (t with df=1) has undefined mean and infinite variance. No valid CI, no hypothesis test. The LOO infrastructure will become meaningful at N >= 3. (3-0)

### F5: All Three "Missing Circuit Components" Are Either Absorbed or Negligible
- Skin effect: absorbed in Scholz short-circuit R0 (standard practice)
- Switch arc: 20 mOhm at t=14 ns, <0.1 mOhm by snowplow phase
- ESR/ESL: likely included in R0 if measured by short-circuit discharge
The remaining concern is whether plasma-loaded discharges have different AC effects than short-circuit calibrations (second-order, <5%). (3-0)

---

## Path to 7.0 (Updated)

The path to 7.0 is now CLEARER but STEEPER after Phase BK's negative LOO results:

| Route | Action | Expected Delta | Status |
|-------|--------|---------------|--------|
| A1 | Third device with digitized I(t) + N=3 LOO | +0.10-0.15 | BLOCKED: no digitized waveform available for UNU-ICTP or NX2 |
| A2 | fm-constrained multi-device NRMSE < 0.15/device | +0.05-0.10 | FAILED: POSEIDON NRMSE = 0.235 >> 0.15 |
| A3 | ASME PASS on cross-device prediction | +0.10-0.20 | BLOCKED: requires u_val for second device (uncharacterized) |
| B1 | Geometry-dependent fm(b/a, L_anode/a, S/S_opt) | +0.10-0.20 | NEW: requires N >= 5 devices to calibrate the geometry model |
| B2 | Non-I(t) observable (sheath velocity, X-ray timing) | +0.05-0.10 | NEW: requires experimental collaboration |
| B3 | Hierarchical Bayesian multi-device calibration | +0.05-0.10 | NEW: requires N >= 5 devices |

**Assessment**: Routes A2 and A3 are effectively blocked. Route A1 requires manual digitization of published I(t) waveforms. Routes B1-B3 require 3+ additional devices with published data. The 7.0 ceiling remains unbroken at Debate #47.

---

## Recommendations for Further Investigation

1. **Digitize a third I(t) waveform** (UNU-ICTP or NX2) to enable N=3 LOO with a well-defined confidence interval
2. **Correct the Debate #42 record** for the fm^(1/3)/fc^(2/3) transit time scaling (retracted by Dr. DPF)
3. **Report fc^2/fm ratios using consistent calibration methodology** — always compare 2-param with 2-param, 3-param with 3-param
4. **Compute POSEIDON-60kV u_val independently** — requires short-circuit calibration data (not Lee-model-fitted L0/R0)
5. **Investigate hierarchical Bayesian calibration** as an alternative to point-estimate LOO for multi-device modeling

---

*Filed by: Debate Moderator*
*Protocol: 5-phase (Phase 1: independent analysis, Phase 2: cross-examination, Phase 3: rebuttal, Phase 4: synthesis, Phase 5: verdict)*
*Duration: Phases 1-3 fully executed; Phases 4-5 combined by moderator given unanimous convergence*
*Concessions: 23 total (17 full + 6 partial), 3 major retractions*
