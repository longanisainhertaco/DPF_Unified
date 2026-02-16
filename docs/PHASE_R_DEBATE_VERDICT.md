# PhD Debate Panel Verdict: Phase R Assessment

**Date**: 2026-02-16
**Panel**: Dr. PP (Pulsed Power), Dr. DPF (Plasma Physics), Dr. EE (Experimental Validation)
**Protocol**: 5-phase debate (Independent Analysis → Cross-Examination → Rebuttal → Synthesis → Verdict)

---

## VERDICT: MAJORITY (3-0 on MHD, 3-0 on DPF-specific)

### Question
Review Phase R work quality, verify bug fixes, update fidelity rating, and assess Phases S-V roadmap.

### Answer

**Phase R delivered correct individual fixes but did not materially change the simulator's ability to model DPF discharges.** The MHD solver infrastructure (Metal backend) is genuinely competitive with established open-source codes. The DPF-specific physics integration remains critically incomplete.

---

## Bifurcated Rating (Consensus)

| Component | Pre-Phase R | Post-Phase R | Justification |
|-----------|-------------|--------------|---------------|
| Metal MHD solver (ideal MHD) | 8.5/10 | **8.5/10** | WENO5-Z + HLLD (confirmed correct) + SSP-RK3 + CT + float64. HLLD double-star states verified present — previous "HLLD=HLLC" bug was false positive. |
| Transport coefficients | 7/10 | **8/10** | Alpha(Z) Braginskii correction correctly applied, unified Coulomb log, deuterium mass fix. Minor: T_eq wrong for Z>1. |
| Circuit solver | 5/10 | **6/10** | Energy split correct, implicit midpoint sound. Missing: crowbar, stray inductance decomposition. |
| DPF-specific physics | 2/10 | **3/10** | Ablation exists but disconnected. No snowplow in MHD engine. No beam-target. No LHDI. Lee model has 2pi bug. |
| Experimental validation | 0/10 | **1/10** | Infrastructure exists (~1500 LOC) but zero end-to-end execution. Lee model cross-check has 2pi force bug. LLNL-DPF data 26x inconsistent. |
| **Composite DPF Simulator** | **4.5/10** | **5.0/10** | Marginal improvement. Correct fixes to transport and energy accounting, but fundamental coupling gaps persist. |

---

## Phase R Fix Quality Assessment

### R.1 — Circuit-Plasma Coupling Fixes: CORRECT (7/10)
- Energy split (external vs plasma Ohmic): **Correct.** Eliminates double-counting.
- back_emf = 0: **Correct.** All three panelists confirm dL/dt enters via R_star. Mathematically equivalent to Lee model.
- R_plasma cap 1000 Ohm: **Acceptable** as safety bound. Should be ~10 Ohm for physical realism (1-line fix).
- dL/dt finite-difference: First-order backward difference introduces ~5-10% timing error during rapid compression (Dr. EE).

### R.2 — Python Engine Demotion: APPROPRIATE (8/10)
- Demoting dp/dt engine to "teaching" rather than rewriting: **Correct decision.**
- Metal engine IS conservative (8-component). Athena++ IS conservative.
- Key gap: Metal engine lacks cylindrical geometry (1/r force scaling), which is MORE important for DPF than energy formulation.

### R.3 — Metal HLLD Double-Star States: VERIFIED CORRECT (9/10)
- **All three panelists independently verified** the implementation at metal_riemann.py:991-1069.
- Miyoshi & Kusano Eqs. 51, 59-60, 62 correctly transcribed.
- Bn→0 degeneracy handled by HLLC fallback.
- 6-region flux selection present.
- **The #1 "surviving confirmed bug" from the 2026-02-14 assessment (HLLD = HLLC) is a FALSE POSITIVE.** Now retracted.

### R.4 — Transport Physics: CORRECT (8/10)
- Alpha(Z) Braginskii: Correctly divides classical eta by alpha(Z), INCREASING resistivity. Direction verified.
- Unified Coulomb log: Gericke-Murillo-Schlanges with quantum diffraction. Correct.
- T-relaxation mass m_p→m_d: Correct for deuterium.
- Minor bug found: T_eq = 0.5*(Te+Ti) should be (Z*Te+Ti)/(Z+1) for Z>1.

---

## Confirmed Bugs (Post-Debate)

### Critical
1. **Lee model force 2π× too large** — `lee_model_comparison.py:297` uses `μ₀/2` instead of `μ₀/(4π)`. The comment on line 290 includes `/(2*pi)` but the code omits it. Internally inconsistent with radial phase (line 345) which correctly uses `μ₀/(4π)`. Confirmed by all three panelists independently. **Impact**: Snowplow validation cross-check produces physically incorrect results (sheath 2.5× too fast).

2. **No snowplow dynamics in MHD engine** — The engine computes R_plasma and L_plasma post-hoc from MHD state, but has no snowplow force driving the current sheet. Cannot produce the current dip. **Impact**: Cannot validate against any experimental I(t) waveform.

### High
3. **Missing LHDI anomalous resistivity** — Only Buneman threshold implemented. LHDI is often dominant in DPF current sheaths (operates at density gradient, lower threshold than Buneman). Ref: Haines (2011) PPCF 53:093001.

4. **Ablation module disconnected** — 228 LOC at `atomic/ablation.py` with complete physics (rate, source, particle flux, momentum). Never called from `engine.py`. Dead code.

5. **No crowbar model** — Affects post-4μs simulation accuracy. Crowbar fires BEFORE pinch on PF-1000 (~4μs vs ~6.5μs pinch). Corrupts energy budget during radial compression.

6. **LLNL-DPF 26× energy inconsistency** — `presets.py`: C=360μF, V0=24kV → 104 kJ. `suite.py`: C=16μF, V0=22kV → 3.9 kJ. Same device name, completely different parameters. Validation meaningless.

### Moderate
7. **T_eq formula wrong for Z>1** — `spitzer.py:260` uses `0.5*(Te+Ti)` regardless of Z. Should be `(Z*Te+Ti)/(Z+1)`. Affects multi-Z plasmas (e.g., Cu ablation).

8. **Non-conservative pressure in Python engine** — dp/dt formulation. Mitigated by demotion to "teaching" tier. ~13% post-shock pressure error at Mach 2.

9. **Missing cylindrical geometry in Metal engine** — J×B force scales as 1/r, which is fundamental to z-pinch physics. Metal engine is Cartesian-only. CylindricalMHDSolver has geometry but lower-order numerics.

### Minor
10. **PF-1000 anode radius inconsistency** — 57.5mm (suite.py, presets.py) vs 58.0mm (experimental.py). 0.9% difference.

11. **Braginskii kappa coefficient Z-dependent** — Uses 3.16 (Z=1 only). Z>1 values slightly different (3.14 for Z=2, 3.12 for Z=3). ~5% error.

---

## Retracted Bugs (Total: 8 of original 20 + 1)

| Bug | Reason for Retraction |
|-----|----------------------|
| B1: BREM_COEFF 10⁸× | Already correct at 1.42e-40 SI |
| B2: Lp 66,000× | Formula correct, error was ~1.3-10× (now fixed) |
| B3: NX2 C 31× | 28μF correct per Lee (2014) |
| B4: Nernst missing dx | dx IS present in np.gradient |
| B5: PF-1000 L0 2.2× | Different measurement conditions |
| B6: Radiation m_p→m_d | m_d IS correct for deuterium |
| B7: V0 mismatch | Fixed in Phase 2 (27kV) |
| **B8: Metal HLLD = HLLC** | **Double-star states confirmed present by all 3 panelists** |

---

## Roadmap Assessment

### Phase S Priority (DPF Physics): CORRECT but REORDER

All three panelists agree snowplow/sheath dynamics (S.1) is the single highest-priority item. Without it, no experimental validation is possible. Recommended reordering:

1. **S.1 Snowplow/sheath** (CRITICAL — blocks all validation)
2. **S.1a Fix Lee model 2π bug** (1-line fix, prerequisite for cross-checking)
3. **S.1b Wire ablation module** (integration, not implementation)
4. **S.1c Add crowbar model** (~50 LOC in rlc_solver.py)
5. **S.4 Cylindrical geometry for Metal** (enables correct 1/r force scaling)
6. **S.2 Beam-target neutrons** (important but not blocking validation)
7. **S.3 Electrode ablation** (already implemented, needs wiring)

### Phase T (Validation): ESSENTIAL but depends on S.1

Phase T cannot meaningfully start until snowplow dynamics produce a current dip. The I_peak within 20% target is achievable; pinch timing within 25% requires the crowbar model. MMS tests (T.3) can proceed independently.

### Phase U (AI/ML): DEFERRED

All three panelists agree AI/ML activation should be deferred until Phase T establishes baseline validation. Training WALRUS on unvalidated simulation data would propagate systematic errors into the surrogate model.

### Phase V (GUI): PARALLEL

Low priority but no dependencies on S/T. Can proceed in parallel.

---

## Panel Positions

- **Dr. PP (Pulsed Power)**: AGREE — "The MHD solver core is technically competent at 7-8/10. The DPF-specific physics stack is at 3-4/10 due to disconnected modules. The pulsed power engineering aspects are 4-5/10 — the implicit midpoint integrator is well-implemented but device parametrization needs work."

- **Dr. DPF (Plasma Physics)**: AGREE — "The codebase contains a genuinely capable MHD solver (Metal WENO5-Z + HLLD + SSP-RK3 at 7.5/10) wrapped around incomplete DPF-specific physics integration (3.0/10). The gap between solver quality and application quality is the defining characteristic."

- **Dr. EE (Experimental Validation)**: AGREE — "From an experimental validation standpoint, the code cannot reproduce the primary diagnostic observable of a DPF (the current waveform shape including the characteristic dip). The circuit model is mathematically sound but physically decoupled from the MHD."

### No Dissenting Opinion

All three panelists converge on the bifurcated assessment: strong MHD solver, weak DPF integration.

---

## Consensus Verification Checklist

- [x] Mathematical derivation provided — Lee model force, Spitzer eta, Lp formula all derived from first principles
- [x] Dimensional analysis verified — R_plasma, Lp, Spitzer eta, anomalous eta all dimensionally checked
- [x] 3+ peer-reviewed citations — Miyoshi & Kusano (2005), Braginskii (1965), Lee & Saw (2014), Scholz et al. (2006), Haines (2011), Borges et al. (2008)
- [x] Experimental evidence cited — PF-1000 I_peak=1.87MA at 27kV (Scholz 2006), neutron yields (Gribkov 2007)
- [x] All assumptions explicitly listed — See individual panelist analyses
- [x] Uncertainty budget — Lee model 2π error quantified, dL/dt timing ~5-10%, L0 uncertainty ~15%
- [x] All cross-examination criticisms addressed — Each panelist responded to every criticism with CONCEDE/DEFEND/PARTIALLY CONCEDE
- [x] No unresolved logical fallacies — EE's "0/10 validation" circular reasoning addressed and revised
- [x] Explicit agreement from each panelist — 3-0 consensus on bifurcated rating

---

## Recommendations for Further Investigation

1. **Immediate (1-day fixes)**:
   - Fix Lee model force: `μ₀/(4π)` not `μ₀/2` (line 297)
   - Fix T_eq for Z>1: `(Z*Te+Ti)/(Z+1)` (line 260)
   - Reconcile LLNL-DPF parameters between suite.py and presets.py
   - Reduce R_plasma cap to 10 Ohm
   - Fix PF-1000 anode radius to 57.5mm consistently

2. **Short-term (1-2 weeks)**:
   - Wire ablation module into engine.py
   - Add crowbar model to rlc_solver.py
   - Run and validate PF-1000 circuit-only simulation against Scholz (2006) I(t)

3. **Medium-term (4-8 weeks)**:
   - Implement snowplow mass-sweep source terms in engine
   - Add cylindrical coordinates to Metal solver
   - Implement LHDI anomalous resistivity

4. **Validation gate**:
   - PF-1000 I(t) waveform within 20% of Scholz et al. (2006)
   - Current dip visible in simulation output
   - Grid convergence study (Richardson extrapolation) for pinch radius
