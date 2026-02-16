# PhD Debate Assessment #4 — Post-Phase S Verdict

**Date**: 2026-02-16
**Panel**: Dr. PP (Pulsed Power), Dr. DPF (Plasma Physics), Dr. EE (Electrical Engineering)
**Protocol**: 5-phase debate (Independent Analysis → Cross-Examination → Rebuttal → Synthesis → Verdict)

---

## VERDICT: CONSENSUS (3-0)

### Composite Score: 5.2/10 (up from 5.0 in Debate #3)

| Component | Score | Trend | Notes |
|-----------|-------|-------|-------|
| MHD Numerics | 7.5/10 | = | WENO-Z + HLLD + SSP-RK3 unchanged |
| Transport | 7.5/10 | = | Braginskii kappa(Z) minor improvement |
| Circuit | 6.0/10 | down | L_coeff bug confirmed, missing f_c |
| DPF-Specific | 3.3/10 | up | Snowplow added but has 2 confirmed bugs |
| Validation | 2.3/10 | up | Infrastructure improved, zero experimental comparison |
| AI/ML | 3.0/10 | = | WALRUS implemented but untrained on DPF data |
| Software Eng. | 7.2/10 | up | 1607+ tests, good discipline |

**Characterization: Bifurcated** — Strong generic MHD infrastructure (7+) but weak DPF-specific physics and validation (2-3). A good MHD code that is not yet a good DPF code.

---

## Confirmed Bugs (6 total)

### Critical (blocks DPF identity)

**Bug 1: L_coeff = F_coeff (snowplow.py:79)**
- Inductance underestimated by exactly 2x
- Should be: `self.L_coeff = 2.0 * self.F_coeff`
- Derivation: L/z = mu_0/(2*pi) * ln(b/a), F/I^2 = mu_0/(4*pi) * ln(b/a) → L_coeff = 2 * F_coeff
- Independently confirmed by all 3 panelists
- Impact: dL/dt 50% too small → current dip too shallow → wrong I(t) shape
- **Fix: 1-line change, immediate priority**

**Bug 2: Missing current fraction f_c in snowplow force**
- Code uses F = F_coeff * I^2, should be F = F_coeff * (f_c * I)^2
- f_c ~ 0.7 (Lee model standard), force overestimated by 1/f_c^2 ~ 2.04x
- Lee model comparison module (lee_model_comparison.py:296) uses f_m*I correctly
- Impact: sheath moves too fast, rundown time underestimated
- **Fix: Add f_c parameter (default 0.7), 5-line change**

**Bug 3: No radial compression phase**
- Only axial rundown (Lee Phase 2) implemented
- No radial implosion, reflected shock, or slow compression (Phases 3-5)
- Cannot produce: pinch, current dip, neutron yield, X-ray burst
- Lee model comparison code HAS a basic radial phase — not integrated into engine
- **Fix: Substantial implementation, Phase T priority**

### High

**Bug 4: Missing LHDI anomalous resistivity**
- Only Buneman threshold implemented in anomalous.py
- LHDI dominates during radial compression
- Needed for radial phase physics

**Bug 5: Ablation module disconnected**
- 228 LOC at atomic/ablation.py, never called by any solver

### Moderate

**Bug 6: Missing cylindrical geometry in Metal engine**
- 1/r force scaling absent in Metal GPU solver
- Blocks Metal for production DPF runs

---

## Retracted Claims (false positives, 3 total)

| Claim | Claimant | Retraction Reason |
|-------|----------|-------------------|
| Cathode b=100mm for PF-1000 | Dr. EE | No source found; all code uses 80mm consistently |
| ESR as significant missing parameter | Dr. PP | Pulse-rated caps at 24 kHz: ESR ~0.025 mOhm, negligible |
| Velocity-Verlet order reduction | Dr. DPF | Scheme IS 2nd-order for smooth v-dependent forces |

---

## Key Debate Findings

### Bug Cancellation Discovery (Dr. DPF, Phase 2)
The L_coeff bug (0.5x) and missing f_c (force 2.04x too high) create a dangerous near-cancellation. The sheath moves ~2x too fast due to excessive force, but L_plasma accumulates 2x too slowly per unit length. Initial claim: L_code/L_correct ~ 1.02 (near-perfect cancellation). **Revised in Phase 3**: L_code/L_correct ~ 0.71 (partial cancellation). This is the "most dangerous class of software bug" — it produces approximately plausible results for doubly-wrong physical reasons.

### Validation Gap (all 3 panelists)
- 148 Phase S tests are 100% unit/integration tests
- validate_current_waveform() exists but never called with simulation output
- No I(t) waveform has ever been compared to experimental data
- No current dip has ever been demonstrated in any simulation
- PF-1000 I_peak inconsistency: experimental.py 1.87 MA vs suite.py 1.80 MA
- ExperimentalDevice dataclass has no measurement uncertainty fields

### Other Verified Points
- BDF2 dL/dt is cosmetic during axial phase but needed during future radial phase
- Voltage reversal 49% wrong for DPF (ignores crowbar + plasma load + pulse cap rating)
- Lee model docstring (line 14) still has pre-R.5 formula (mu_0/2 instead of mu_0/(4pi))
- LHDI threshold implementation is correctly established in literature (Davidson & Gladd 1975)

---

## Panel Positions

- **Dr. PP: AGREE** — Snowplow is a step forward, but L_coeff bug and missing f_c mean quantitatively wrong dynamics. Circuit drops to 5.5 due to confirmed errors. 5.2 is fair.
- **Dr. DPF: AGREE** — Right architecture but two real bugs. Missing radial phase is the elephant in the room. 5.2 consistent with component scores.
- **Dr. EE: AGREE** — Validation infrastructure exists but never exercised. 148 tests are structurally valuable but physically shallow. L_coeff bug is elementary and should have been caught. 5.2 is appropriate.

---

## Recommendations for Phase T (Priority-Ordered)

1. **Fix L_coeff = 2 * F_coeff** — 1-line fix, immediate. Impact: +0.3 on Circuit
2. **Add current fraction f_c** — 5-line fix, immediate. Impact: +0.2 on Circuit
3. **Implement radial compression phase** — Highest-impact feature. Impact: +2.0 on DPF-Specific
4. **Run first experimental validation** — Compare I(t) vs Scholz et al. PF-1000 data. Impact: +1.5 on Validation
5. **Add LHDI anomalous resistivity** — Needed for radial phase. Impact: +0.5 on DPF-Specific
6. **Fix Lee model docstring** — Trivial fix
7. **Add cylindrical geometry to Metal engine** — Enables Metal for production DPF runs
8. **Connect ablation module** — Low priority until radial phase exists

### Projected Post-Phase-T Composite (if items 1-4 completed): 6.3/10

---

## Assessment History

| Assessment | Composite | Key Change |
|------------|-----------|------------|
| Debate #2 (Phase R) | 5.0/10 | "No snowplow dynamics" |
| Debate #3 (pre-Phase S) | 5.0/10 | 8 bugs retracted, 11 confirmed |
| **Debate #4 (Phase S)** | **5.2/10** | Snowplow added but buggy, 3 claims retracted |

---

## Consensus Verification Checklist

- [x] Mathematical derivation provided (coaxial inductance from magnetic energy)
- [x] Dimensional analysis verified (all formulas SI-consistent)
- [x] 3+ peer-reviewed citations (Lee & Saw 2014, Scholz 2006, Davidson & Gladd 1975, Auluck 2021)
- [x] Experimental evidence cited (PF-1000 I_peak = 1.87 MA, Scholz et al.)
- [x] All assumptions explicitly listed (thin sheath, f_c, MHD validity conditions)
- [x] Uncertainty budget (composite +/- 0.5 points, experimental I_peak +/- 9%)
- [x] All cross-examination criticisms addressed (Phase 3 rebuttals complete)
- [x] No unresolved logical fallacies (3 retracted, all remaining claims supported)
- [x] Explicit agreement from each panelist (3-0 consensus)
