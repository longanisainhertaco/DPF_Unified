# PhD Debate Panel Verdict #3: Full DPF-Unified Project Assessment

**Date**: 2026-02-16
**Panel**: Dr. PP (Pulsed Power), Dr. DPF (Plasma Physics), Dr. EE (Experimental Validation)
**Protocol**: 5-phase debate (Independent Analysis → Cross-Examination → Rebuttal → Synthesis → Verdict)
**Scope**: Entire DPF-Unified project (not just a single phase)

---

## VERDICT: CONSENSUS (3-0) — Composite Score: 4.5/10

### Question
Is the self-assessed 8.9/10 fidelity rating for DPF-Unified honest? What is the true state of the project as a dense plasma focus simulator? What should be prioritized next?

### Answer

**The 8.9/10 self-assessment is significantly inflated. The consensus rating is 4.5/10 as a DPF simulator.** The project contains a genuinely capable general-purpose MHD solver (7.5-8.5/10) wrapped around critically incomplete DPF-specific physics integration (2.0-3.0/10) and zero exercised experimental validation (1.0-2.0/10). The gap between "good MHD code" and "DPF simulator" is the defining characteristic of this project.

The code cannot reproduce the primary diagnostic observable of a DPF discharge — the current waveform with its characteristic dip at pinch time. Without snowplow dynamics driving mass accumulation in the current sheath, no validation against any experimental I(t) measurement is possible.

---

## Bifurcated Rating (3-0 Consensus)

| Component | Rating | Justification |
|-----------|--------|---------------|
| **MHD Numerics** (solver infrastructure) | **7.5/10** | WENO5-Z + HLLD + SSP-RK3 + CT + float64 is competitive with established open-source codes. HLLD double-star states verified correct (Miyoshi & Kusano 2005). Sod/Brio-Wu verification passing. Convergence order ~1.86 (limited by nonlinearity, not scheme). Docked from 8.5 for: Python engine hybrid WENO5 instability, Metal engine Cartesian-only (no cylindrical geometry). |
| **Transport Coefficients** | **7.5/10** | Spitzer with alpha(Z) Braginskii correction, GMS Coulomb log, deuterium mass fix. Braginskii full anisotropic tensor. T_eq fixed for Z>1. Docked for: kappa_e uses Z=1 coefficient only (~5% error at Z>1), anomalous resistivity threshold mislabeled as "Buneman" when actually ion-acoustic (v_d > v_ti, not v_d > v_te). |
| **Circuit Solver** | **5.5/10** | Implicit midpoint method is sound (~1% energy conservation). Energy split (external vs plasma Ohmic) correctly implemented. Docked for: no crowbar model (fires at ~4 μs on PF-1000, BEFORE pinch), no stray inductance decomposition (L₀=33.5 nH is measured total, not computed parasitic budget), first-order backward difference for dL/dt introduces ~5-10% timing error. |
| **DPF-Specific Physics** | **2.0/10** | No snowplow dynamics — the defining physics of a DPF (J×B-driven mass accumulation in current sheath). Ablation module exists (228 LOC) but is disconnected. No LHDI anomalous resistivity. No beam-target neutron production. Metal engine lacks cylindrical geometry (1/r force scaling). Lee model comparison exists but is standalone. |
| **Experimental Validation** | **1.5/10** | Infrastructure exists (~1500 LOC: suite.py, experimental.py, lee_model_comparison.py) but has never been exercised end-to-end. Zero simulation-vs-experiment comparisons produced. PF-1000 reference data present (Scholz 2006: I_peak=1.87 MA at 27 kV). Lee model cross-check had 2π bug (now fixed). Shock tube verification passes but validates MHD numerics, not DPF physics. |
| **AI/ML (WALRUS Integration)** | **3.0/10** | WALRUS 1.3B surrogate is fully implemented (surrogate.py, inference pipeline, RevIN normalization). BUT: trained on unvalidated simulation data — surrogate inherits systematic errors. Confidence module uses "validated" label without validation data. Premature until Phase T establishes baseline. |
| **Software Engineering** | **7.0/10** | Clean architecture (tri-engine with unified base class), Pydantic v2 config, 1494 non-slow tests, CI gate at 745+, Strang splitting, comprehensive device presets. Docked for: dormant modules (AMR 755 LOC, PIC 978 LOC, multi-species 409 LOC) are code debt, not features. |
| **COMPOSITE** | **4.5/10** | Weighted: MHD Numerics 15% + Transport 10% + Circuit 15% + DPF-Specific 30% + Validation 20% + AI/ML 5% + Software 5% |

### Weighting Rationale
DPF-specific physics (30%) and validation (20%) together account for 50% because **the project's stated purpose is DPF simulation**. An MHD solver without snowplow, crowbar, or cylindrical geometry is a general-purpose MHD code, not a DPF simulator.

---

## Debate Record Summary

### Phase 1: Independent Analysis (Parallel)
- **Dr. PP**: 3.5/10 — Circuit solver correct but missing crowbar, snowplow, parasitic budget
- **Dr. DPF**: 4.8/10 — Good MHD solver (8.5) but DPF-specific physics absent (2.0)
- **Dr. EE**: 1.5/10 — Zero end-to-end validation runs, initially characterized as "Potemkin village"

### Phase 2: Cross-Examination (Key Catches)
- Dr. EE caught Dr. PP's arithmetic error (T/4 = 663 μs wrong, used L₀=33.5e-6 H instead of 33.5e-9 H; correct T/4 = 10.49 μs)
- Dr. EE identified code's "Buneman" is actually ion-acoustic (v_d > v_ti, not v_d > v_te)
- Dr. PP challenged Dr. DPF's ALEGRA parity claim and Lee model undervaluation
- Dr. DPF challenged Dr. EE's "Potemkin village" characterization as unfair given real engineering quality

### Phase 3: Rebuttal (Score Revisions)
- Dr. PP: Conceded T/4 error, revised 3.5 → 4.1/10
- Dr. DPF: Conceded LHDI threshold imprecision (0.14 → 0.129 for deuterium), anomalous resistivity relabeling. Revised 4.8 → 5.0/10
- Dr. EE: Withdrew "Potemkin village," conceded PF-1000 data within shot-to-shot variation (±5-10%). Replaced ASME V&V 20-2009 standard as inappropriate for research code. Revised 1.5 → 2.0/10

### Phase 4: Synthesis (Convergence)
- Dr. DPF proposed 4.5/10 with explicit component weighting
- Dr. PP proposed two-tier system (6.5 as MHD, 3.5 as DPF), compromise 4.5/10
- Dr. EE converged to 4.5/10 acknowledging infrastructure credit
- All three independently arrived at 4.5/10

---

## Supporting Evidence

### Mathematical Verification

**1. Snowplow Force (Missing)**
The snowplow equation for the axial rundown phase (Lee model Phase 2):

    d/dt[m(z) * dz/dt] = (μ₀/4π) * ln(b/a) * I² - pressure terms

where m(z) = ρ₀ × π(b² - a²) × z × f_m is the swept mass. This term does NOT exist in the MHD engine. The code computes R_plasma and L_plasma from the MHD state post-hoc but applies no snowplow force.

**2. Circuit Quarter-Period**
T/4 = (π/2)√(L₀C) = (π/2)√(33.5×10⁻⁹ × 360×10⁻⁶) = 10.49 μs for PF-1000

**3. Anomalous Resistivity Threshold**
- Code implements: v_d > v_ti (ion-acoustic-like, Sagdeev 1966)
- True Buneman: v_d > v_te (Buneman 1959)
- LHDI: v_d > (m_e/m_i)^{1/4} × v_ti ≈ 0.129 × v_ti for deuterium
- Code threshold is ~8× above LHDI threshold — anomalous resistivity activates too late

**4. HLLD Verification**
Metal HLLD (metal_riemann.py:991-1069) independently verified against Miyoshi & Kusano (2005) Eqs. 51, 59-60, 62 by all three panelists. Double-star states present. Bn→0 degeneracy handled by HLLC fallback. Previous "HLLD=HLLC" bug was a FALSE POSITIVE (retracted).

### Literature Citations

1. **Lee, S. & Saw, S.H.** (2014). "Plasma focus ion beam fluence and flux—Scaling with stored energy." *Physics of Plasmas*, 21(7), 072501. DOI: 10.1063/1.4890971
2. **Miyoshi, T. & Kusano, K.** (2005). "A multi-state HLL approximate Riemann solver for ideal MHD." *J. Comput. Phys.*, 208(1), 315-344. DOI: 10.1016/j.jcp.2005.02.017
3. **Scholz, M. et al.** (2006). "PF-1000 device operation with various gas conditions." *Nukleonika*, 51(2), 79-84.
4. **Haines, M.G.** (2011). "A review of the dense Z-pinch." *Plasma Phys. Control. Fusion*, 53(9), 093001. DOI: 10.1088/0741-3335/53/9/093001
5. **Borges, R. et al.** (2008). "An improved WENO scheme for hyperbolic conservation laws." *J. Comput. Phys.*, 227(6), 3191-3211. DOI: 10.1016/j.jcp.2007.11.038
6. **Shu, C.-W. & Osher, S.** (1988). "Efficient implementation of ENO shock-capturing schemes." *J. Comput. Phys.*, 77(2), 439-471. DOI: 10.1016/0021-9991(88)90177-5

---

## Assumptions and Limitations

1. **Ideal MHD approximation** — valid for β ≲ 1, Rm >> 1, Kn << 1. DPF pinch phase: β ~ 0.01-0.1 (valid), Rm ~ 10-100 (marginal), Kn ~ 0.01-0.1 (marginal).
2. **Cartesian geometry on Metal engine** — DPF is fundamentally cylindrical (r, z). J×B force scales as 1/r. Metal engine operates only in Cartesian coordinates.
3. **No mass accumulation model** — Engine does not sweep up gas ahead of the current sheath. Conceptual gap.
4. **Lumped-element circuit** — Sufficient for kJ-class devices. Breaks down for MJ-class (MITL transit-time effects).
5. **Single-fluid MHD** — Cannot resolve kinetic instabilities (m=0 sausage, m=1 kink) during final compression.

---

## Uncertainty Budget

| Quantity | Claimed | Actual Uncertainty | Source |
|----------|---------|-------------------|--------|
| Self-assessed fidelity | 8.9/10 | **4.5 ± 0.5 /10** | Panel consensus |
| Energy conservation | ~1% | Reasonable for implicit midpoint | Circuit solver verified |
| Spitzer resistivity | Within 5% of NRL Formulary | ~10% at Z>1 due to kappa approx | Dr. DPF analysis |
| HLLD accuracy | Correct | Confirmed by all 3 panelists | Direct code inspection |
| PF-1000 I_peak prediction | Not attempted | Cannot predict without snowplow | Fundamental gap |
| Convergence order | ~1.86 | Correct (nonlinearity limited) | Phase O tests |
| PF-1000 experimental I_peak | 1.87 MA (Scholz 2006) | ±5-10% shot-to-shot | Dr. EE analysis |

---

## Panel Positions

### Dr. PP (Pulsed Power): AGREE — 4.5/10
"The circuit model is mathematically sound — implicit midpoint with correct energy split is good engineering. But the pulsed power aspects are incomplete. No crowbar model means the energy budget is wrong after ~4 μs on PF-1000. The L₀=33.5 nH is a measured lump sum, not a decomposed parasitic budget. The gap between circuit solver quality (5.5/10) and DPF-specific physics (2/10) is the fundamental issue. Compromise: 4.5/10 composite."

### Dr. DPF (Dense Plasma Focus): AGREE — 4.5/10
"The MHD solver infrastructure is genuinely competitive. WENO5-Z + HLLD + SSP-RK3 on Metal with float64 and CT would score well against any open-source MHD code. But a DPF simulator must solve DPF physics. Without snowplow dynamics, the code cannot produce the most basic observable — the current dip. Composite 4.5/10 with explicit weighting."

### Dr. EE (Experimental Validation): AGREE — 4.5/10
"From an experimental validation standpoint, this code has never produced a single comparison against a measured waveform. The 1494 tests are predominantly unit tests and MHD verification (Sod, Brio-Wu), which validate the solver, not the physics application. I withdrew my initial 'Potemkin village' characterization — the code has real engineering quality — but the validation gap remains the single largest credibility issue."

### No Dissenting Opinion
All three panelists converge on 4.5/10 through independent calculation methodologies.

---

## Comparison to Previous Assessment

| Metric | PhD Debate #2 (Phase R) | PhD Debate #3 (Full Project) | Change |
|--------|------------------------|------------------------------|--------|
| MHD solver | 8.5/10 | 7.5/10 | -1.0 |
| Transport | 8/10 | 7.5/10 | -0.5 |
| Circuit | 6/10 | 5.5/10 | -0.5 |
| DPF-specific | 3/10 | 2.0/10 | -1.0 |
| Validation | 1/10 | 1.5/10 | +0.5 |
| **Composite** | **5.0/10** | **4.5/10** | **-0.5** |

The slight downward revision reflects deeper scrutiny in this full-project review. Phase R fixes were correct but did not materially change the project's ability to simulate DPF discharges.

---

## Comparison to Reference Codes

| Code | Type | Snowplow | Current Dip | V&V | DPF Score |
|------|------|----------|-------------|-----|-----------|
| **Lee Model (RADPF)** | Lumped circuit, 5-phase | YES | YES | 40+ devices | 7/10 |
| **Chicago/LSP** | PIC/MHD hybrid | YES | YES | Peer-reviewed | 9/10 |
| **ALEGRA** | ALE MHD + rad-hydro | YES (Lagrangian) | YES | Sandia-certified | 9.5/10 |
| **OpenMHD** | Conservative MHD | No (general purpose) | No | Community benchmarks | 5/10 |
| **DPF-Unified** | Tri-engine MHD | **NO** | **NO** | MHD verification only | **4.5/10** |

The Lee Model — a lumped-circuit code with ~2000 LOC — currently outperforms DPF-Unified on DPF predictive capability because it has snowplow dynamics, current dip reproduction, and extensive experimental validation across dozens of devices.

---

## Priority Recommendations

### Immediate (1-week sprint) — Phase S

**S.1: Implement Snowplow Source Terms** (CRITICAL — blocks everything)
- Add mass-sweep source term to engine.py
- This single item transforms "MHD code" into "DPF simulator"

**S.2: Wire Ablation Module** (HIGH — 228 LOC already written)
- Connect atomic/ablation.py to engine.py operator-split loop

**S.3: Add Crowbar Model** (HIGH — ~50 LOC)
- Add crowbar switch logic to rlc_solver.py
- Fires at V_cap → 0, short-circuits capacitor

### Short-term (2-4 weeks) — Phase T

**T.1: First Experimental Validation Run** (CRITICAL)
- Run PF-1000 simulation end-to-end
- Compare I(t) against Scholz (2006): I_peak within 20%

**T.2: Lee Model Cross-Check** (HIGH)
- Run DPF-Unified and Lee model side-by-side

**T.3: Fix Anomalous Resistivity** (MODERATE)
- Rename "Buneman" to "ion-acoustic"
- Add LHDI threshold: v_d > (m_e/m_i)^{1/4} × v_ti

### Medium-term (4-8 weeks) — Phase U

**U.1: Cylindrical Geometry for Metal Engine** (HIGH)
**U.2: Grid Convergence Study** (MODERATE)
**U.3: WALRUS Retraining** (LOW — defer until T.1 passes)

---

## Consensus Verification Checklist

- [x] Mathematical derivation provided
- [x] Dimensional analysis verified
- [x] 3+ peer-reviewed citations (6 with DOIs)
- [x] Experimental evidence cited (PF-1000, Scholz 2006)
- [x] All assumptions explicitly listed (5, with regimes)
- [x] Uncertainty budget (7 quantities)
- [x] All cross-examination criticisms addressed
- [x] No unresolved logical fallacies
- [x] Explicit agreement from each panelist (3-0)

---

## The Identity Gap

The single most important finding: there is a fundamental **identity gap** between what the code is (a general-purpose MHD solver with DPF device presets) and what it claims to be (a dense plasma focus simulator rated 8.9/10).

Closing this gap requires:
1. **Snowplow mass accumulation** — axial rundown current sheath sweeps up gas
2. **Cylindrical J×B compression** — 1/r force scaling drives radial implosion
3. **Crowbar switch** — circuit event dominating energy budget during pinch

Until these are implemented and validated against at least one experimental I(t) waveform, the honest rating remains **4.5/10 as a DPF simulator** (or equivalently: 7.5/10 as a generic MHD code).
