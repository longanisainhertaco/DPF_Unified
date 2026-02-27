# PhD Debate #15 Verdict — Phase U Architecture Migration Proposal

**Date**: 2026-02-27
**Score**: 6.7/10 (NO CHANGE from Debate #14)
**Delta**: 0.0

## VERDICT: MAJORITY (2-1)

### Question
Assess the Phase U Architecture Migration Proposal: evaluate the scientific credibility of the proposed architecture changes (Metal→MLX, WALRUS→FNO, AthenaK deferred, Athena++ primary) and the project's current state.

### Answer
The Phase U Architecture Migration Proposal correctly identifies coupling validation as the #1 priority (unanimous 3-0). The debate produced 12+ significant concessions across all panelists, correcting numerous factual errors from Phase 1 analyses. However, no new code, validation evidence, or physics capabilities were demonstrated. The score remains at 6.7/10. The hard ceiling at 7.0 requires a coupled MHD+circuit PF-1000 simulation producing I(t) compared against Scholz (2006) experimental data.

### Consensus Verification Checklist

- [x] **Mathematical derivation provided** -- RLC analytical benchmark (omega_0, I_peak, Q factor), Kadomtsev sausage formula (gamma ~ v_A/a), collisionless skin depth (c/omega_pe = 1.7 um)
- [x] **Dimensional analysis verified** -- BREM_COEFF [W m^3 K^{-1/2}], CFL dt [s], NRMSE [dimensionless]
- [x] **3+ peer-reviewed citations** -- Scholz et al. (2006), Lee & Saw (2014), Miyoshi & Kusano (2005), Kadomtsev (1966), Haines (2011), AIAA G-077-1998
- [x] **Experimental evidence cited** -- Scholz (2006) PF-1000 I(t) waveform, NRMSE=0.1329
- [x] **All assumptions explicitly listed** -- 8+ per panelist with regime of validity
- [x] **Uncertainty budget** -- Sausage range 3-75 ns, float32 ~1e-7, I_peak 14% (k=1), CFL 0.07-0.23 ns
- [x] **All cross-examination criticisms addressed** -- 12+ concessions documented
- [ ] **No unresolved logical fallacies** -- Dr. EE dual-axis scoring methodology differs from established sub-score framework
- [x] **Explicit agreement/dissent** -- Dr. PP AGREE 6.8, Dr. DPF AGREE 6.7, Dr. EE DISSENT 6.2

### Score Progression
| Debate | Score | Delta | Key Driver |
|--------|-------|-------|-----------:|
| #12 | 6.4 | +0.1 | Phase AD engine validation + strategy debate |
| #13 | 6.5 | +0.1 | HLL momentum_flux bug fix + PF-1000 grid margin |
| #14 | 6.7 | +0.2 | Validation benchmark suite + Sedov tightening + HLLD XPASS |
| **#15** | **6.7** | **0.0** | **Phase U Architecture debate -- concessions only, no new evidence** |

### Sub-Score Breakdown (Unchanged from Debate #14)

| Category | Debate #14 | Debate #15 | Change | Justification |
|----------|-----------|-----------|--------|---------------|
| MHD Numerics | 8.2 | 8.2 | 0.0 | No new MHD work |
| Transport | 7.7 | 7.7 | 0.0 | Bremsstrahlung confirmed (already credited) |
| Circuit | 6.7 | 6.7 | 0.0 | Crowbar confirmed (already credited) |
| DPF Coupling | 6.2 | 6.2 | 0.0 | Phase U cylindrical verified but not validated |
| Validation | 5.8 | 5.8 | 0.0 | NRMSE=0.1329 confirmed (already credited) |
| AI/ML | 3.5 | 3.5 | 0.0 | WALRUS temporal advantage retraction confirms prior score |
| Software | 7.5 | 7.5 | 0.0 | 270 tests confirmed (correction, not improvement) |

**Composite** (MHD and DPF double-weighted): 6.58 -> **6.7/10**

### Points of Agreement (13 items, 3-0 consensus)

1. **NRMSE = 0.1329** validates circuit+snowplow subsystem only, NOT the MHD solver (HIGH)
2. **270 dedicated tests** (not 395) -- count corrected (HIGH)
3. **Bremsstrahlung is implemented** in both Python and Metal engines with correct SI coefficient (HIGH)
4. **Crowbar model is correctly implemented** -- voltage-zero trigger, post-crowbar L-R decay (HIGH)
5. **Sausage instability timescale**: 3-75 ns range, not a single number (HIGH)
6. **Collisionless skin depth**: 1.7 um at n_e=10^26, not 0.01 mm (HIGH)
7. **WALRUS advantage is architectural**, not temporal (MEDIUM)
8. **FNO Gibbs concern is moot** for grid-resolved training data (MEDIUM)
9. **Float32 cancellation**: negligible at equilibrium, significant at Mach > 100 (HIGH)
10. **Peak I and peak dL/dt do not coincide** (HIGH)
11. **Score range 6.8-6.9** supersedes withdrawn 7.4+/-0.4 (HIGH)
12. **Du et al. (2025)** citation retracted as irrelevant (HIGH)
13. **Coupling validation is #1 priority** -- no full MHD+circuit experimental comparison exists (HIGH)

### Remaining Disagreements (4 items)

| ID | Disagreement | PP | DPF | EE |
|----|-------------|----|----|-----|
| RD1 | Bremsstrahlung operator-split error | Adequate | Needs quantification | Medium concern |
| RD2 | Validation sub-score methodology | 5.8 | 5.4 (restructured) | Tier pyramid |
| RD3 | HLLD vs HLLC-MHD labeling | Accepted | Concern maintained | Accepted |
| RD4 | Overall score | 6.8 | 6.7 | 6.2 |

### Panel Positions

- **Dr. PP (Pulsed Power)**: AGREE at 6.8 -- Conceded 4/10 scholarly rigor (self-assessed). Confirmed crowbar correct via full code review and analytical reversal calculation (48.7% for PF-1000). Notes NX2 preset has no crowbar despite 74.5% voltage reversal at Q=5.3. Awards +0.1 for credibility margin.

- **Dr. DPF (Plasma Physics)**: AGREE at 6.7 -- Maintains error corrections restore baselines, not earn new credit. Made 13 concessions including: bremsstrahlung IS implemented, sausage range 3-75 ns, skin depth 1.7 um, Gibbs moot, WALRUS temporal claim retracted. Most rigorous scoring methodology.

- **Dr. EE (Electrical Engineering)**: DISSENT at 6.2 -- Proposes dual-axis scoring: Axis 1 (numerics) 8.7/10, Axis 2 (DPF physics) 4.0/10, composite 6.2. Introduces validation pyramid: T1 analytical COMPLETE, T2 circuit+snowplow COMPLETE (NRMSE=0.1329), T3 full MHD+circuit NOT DEMONSTRATED. Ran full test suite: 2,627 non-slow pass in 87s.

### Dissenting Opinion (Dr. EE, 6.2/10)

A DPF simulator that cannot demonstrate the current dip from a full MHD-coupled simulation is not a DPF simulator. It is a very well-verified RLC circuit solver attached to an MHD engine that has not yet been shown to influence the circuit in a measurable way. The scoring should reflect this:
- Numerical methods (Axis 1): 8.7/10
- DPF physics fidelity (Axis 2): 4.0/10
- Composite: 6.2/10

The Tier 3 validation gap (full MHD+circuit vs experiment) caps Axis 2 at 4.0 until demonstrated.

### Concessions This Debate (12+ across all panelists)

| Who | Conceded | Claim Corrected |
|-----|----------|----------------|
| Dr. PP | Self-assessed 4/10 scholarly rigor | Multiple citation failures |
| Dr. PP | Du et al. (2025) retracted | Irrelevant to DPF |
| Dr. PP | 750kV retracted | dL/dt temporal inconsistency |
| Dr. PP | v_fast mismatch conceded | Inconsistent with own analysis |
| Dr. DPF | Bremsstrahlung IS implemented | "No radiative collapse pathway" retracted |
| Dr. DPF | Sausage range 3-75 ns | Was citing single value |
| Dr. DPF | Skin depth 1.7 um | Was 0.01 mm (6x error) |
| Dr. DPF | Gibbs concern moot | For grid-resolved data |
| Dr. DPF | WALRUS temporal advantage retracted | Architectural, not temporal |
| Dr. EE | NRMSE 0.503 retracted | Calibrated = 0.1329 |
| Dr. EE | Test count 270, not 395 | Overcounting corrected |
| Dr. EE | Score 7.4+/-0.4 withdrawn | Evidence-based: 6.8-6.9 |

### Key Findings

1. **Validation Pyramid** (Dr. EE, accepted as useful framework):
   - Tier 1: Analytical verification -- COMPLETE (Sod, Brio-Wu, Sedov, Bennett, linear waves)
   - Tier 2: Circuit+snowplow vs experiment -- COMPLETE (NRMSE=0.1329 vs Scholz 2006)
   - Tier 3: Full MHD+circuit vs experiment -- NOT DEMONSTRATED
   - Tier 4: Sub-us dynamics vs experiment -- NOT DEMONSTRATED

2. **CFL at pinch conditions**: 0.07-0.23 ns (Dr. EE computed). 50,000 steps for 5 us PF-1000 simulation.

3. **NX2 crowbar gap**: 74.5% voltage reversal at Q=5.3, no crowbar in preset (Dr. PP computed).

4. **Error corrections != improvements** (Dr. DPF): A debate producing only concessions does not merit a score increase. The net effect on the code is zero.

5. **PF-1000 analytical peak**: 3.927 MA unloaded, Q=2.2, R_crit=10.03 mOhm (Dr. PP/EE computed).

### Recommendations for Further Investigation

1. **Coupled MHD+circuit PF-1000 simulation** (Tier 3 validation) -- highest-leverage action (+0.3-0.5)
2. **Write `test_engine_coupling_current_dip.py`** -- verify current dip in full engine run
3. **Circularly polarized Alfven wave test** -- resolve HLLD vs HLLC-MHD concern
4. **Bremsstrahlung manufactured solution test** -- quantify operator-split error
5. **NX2 crowbar assessment** -- confirm capacitor reversal rating or add crowbar
6. **R0 = 2.3 mOhm citation** -- traceable provenance still missing

### Path to 7.0

| Action | Expected Impact | Status |
|--------|----------------|--------|
| Coupled MHD+circuit PF-1000 vs Scholz I(t) | +0.3-0.5 | BLOCKED (requires production run) |
| Cylindrical cross-backend parity (Metal vs Athena++) | +0.1 | NOT STARTED |
| Circularly polarized Alfven wave at oblique angle | +0.1 | NOT STARTED |
| R0 citation traceability | +0.0 | NOT STARTED |
| NX2 crowbar assessment | +0.0 | NOT STARTED |

### Citations

1. Scholz M. et al., Nukleonika 51(1):79-84 (2006) -- PF-1000 parameters and I(t)
2. Lee S. & Saw S.H., J. Fusion Energy 33:319-335 (2014) -- fc/fm calibration ranges
3. Miyoshi T. & Kusano K., JCP 208:315-344 (2005) -- HLLD Riemann solver
4. Kadomtsev B.B., Reviews of Plasma Physics Vol. 2 (1966) -- sausage instability
5. Haines M.G., PPCF 53:093001 (2011) -- Pease-Braginskii review
6. AIAA G-077-1998 -- Guide for verification and validation in CFD
7. Bruzzone H. & Aranchuk L., J. Phys. D: Appl. Phys. 36:2218 (2003) -- electrode ablation
