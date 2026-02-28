# PhD Debate #23 Verdict: Post-Phase AK Grid Convergence at DPF-Relevant Conditions

## VERDICT: MAJORITY (2-1) — Score: 6.3/10 (unchanged)

### Question
Does Phase AK (grid convergence study at PF-1000 fill-gas conditions: sound wave and fast magnetosonic wave convergence, float32 vs float64 precision comparison, repair fraction diagnostics, R_plasma convergence across 3 resolutions) break the 7.0 ceiling?

### Answer
**No.** Phase AK is a competent verification study that measures convergence order at PF-1000-relevant density and pressure scales. It confirms PLM convergence order 1.30, reveals float32 precision degradation (1.52 to 1.03), and validates that the positivity fallback triggers zero repairs on smooth data. However, it remains a Cartesian-geometry linear wave test — not a DPF simulation. All three panelists agree this is verification, not validation, and that Cartesian tests cannot probe DPF-relevant physics (cylindrical geometry, electrode boundary conditions, radiation). Two panelists (Dr. PP, Dr. EE) hold at 6.3/10; Dr. DPF dissents at 6.4/10, crediting the proven-exact eigenvector and sound methodology with a marginal increment.

### Panel Positions
- **Dr. PP (Pulsed Power):** 6.3/10 — "Phase AK confirms the Metal MHD solver converges at expected rates for smooth linear waves. PLM order 1.30 is credible; WENO5 order 2.14 is unreliable due to the 4-cell transverse grid. Peak current convergence proves snowplow dominance, not MHD convergence. This is verification infrastructure — necessary but not sufficient for 7.0."
- **Dr. DPF (Plasma Physics):** 6.4/10 — "The eigenvector was proven exact for perpendicular propagation (Bx=0) to machine precision. I withdraw my ~10% error claim entirely. The convergence methodology is sound, gamma=5/3 does not affect convergence order, and the coupled test is scope-limited rather than vacuous. These corrections warrant +0.1 over my previous assessment. However, smooth waves are the easiest regime, the pinch phase remains untested, and shocks are needed."
- **Dr. EE (Electrical Engineering):** 6.3/10 — "Phase AK is textbook verification. The fast wave eigenvector is exact for this geometry — I withdraw my 4.56% estimate (from oblique propagation context). Float32 precision degradation is documented, not a deficiency, but Metal GPU cannot run the highest-fidelity configuration. No Cartesian test can validate DPF physics. Verification alone scores zero without a roadmap to validation."

### Concessions (Phase 3)

**Dr. PP (4 full concessions):**
1. CONCEDES 70% voltage reversal was wrong — computed 47.7% independently; PF-1000 crowbar makes it 0% in operation
2. CONCEDES peak current convergence proves snowplow dominance, not MHD convergence
3. CONCEDES Phase AK is verification, not validation
4. CONCEDES WENO5 convergence order 2.14 is unreliable due to quasi-1D grid (4 transverse cells)

**Retains:** PLM order 1.30 credible; float32 degradation (1.52 to 1.03) valuable; zero repairs on smooth data confirms positivity fallback correctness.

**Dr. DPF (4 full concessions):**
1. CONCEDES gamma=5/3 does not invalidate convergence study (order is scheme-dependent, not physics-dependent)
2. CONCEDES fast wave eigenvector IS exact for perpendicular propagation (Bx=0) — verified by full eigenanalysis to machine precision. WITHDRAWS "~10% error" claim entirely.
3. CONCEDES "MHD not applicable at 300K" is a validation objection, not a verification objection — convergence study is valid verification
4. CONCEDES coupled test is "scope-limited, not vacuous" — withdraws word "vacuous"

**Retains:** WENO5 order 2.14 needs explanation; smooth waves are easiest regime; shocks needed; pinch phase untested; gamma labeling should be corrected.

**Dr. EE (6 full concessions):**
1. CONCEDES 6.4 upper bound was generous — withdraws it, score is exactly 6.3
2. CONCEDES float32 limitation is documented, not a deficiency (notes Metal GPU cannot run highest-fidelity config)
3. CONCEDES repair diagnostic credit is premature — zero repairs on smooth waves is expected
4. REJECTS Dr. DPF's eigenvector criticism with full eigenanalysis proof — Phase AK eigenvector is exact (0% error for Bx=0). WITHDRAWS own 4.56% estimate (was from oblique propagation context).
5. CONCEDES no Cartesian test can test DPF-relevant physics (cylindrical geometry, electrode BCs, radiation)
6. CONCEDES verification alone scores zero without roadmap to validation

**Retains:** Float32-only on Metal GPU means highest-fidelity schemes cannot run on target hardware at production resolution; Cartesian tests fundamentally cannot test DPF physics.

### Points of Agreement (All Three Panelists)
1. Phase AK is **verification**, not validation
2. The fast wave eigenvector is **exact** for perpendicular propagation (Bx=0) — all three now agree, prior error estimates withdrawn
3. Gamma=5/3 does not affect convergence order (only labeling)
4. WENO5 order 2.14 is unreliable due to 4-cell transverse grid
5. Peak current convergence proves snowplow dominance, not MHD convergence
6. Smooth linear waves are the easiest regime for convergence
7. Repair diagnostic is infrastructure, not yet tested on relevant problems
8. Dr. PP's 70% voltage reversal was wrong (actual 47.7%, or 0% with crowbar)

### Sub-Score Impact

| Subsystem | Score | Delta | Rationale |
|-----------|-------|-------|-----------|
| MHD Numerics | 8.0 | 0.0 | PLM order 1.30 confirmed at DPF scales; WENO5 order unreliable |
| Transport Physics | 7.5 | 0.0 | No changes |
| Circuit Solver | 6.5 | 0.0 | No changes |
| DPF-Specific Physics | 5.8 | 0.0 | Cartesian test adds no DPF physics |
| Validation & V&V | 4.8 | 0.0 | Verification only, no new experimental comparison |
| AI/ML Infrastructure | 4.0 | 0.0 | No changes |
| Software Engineering | 7.5 | 0.0 | Clean test implementation, no architectural change |

### Consensus Verification Checklist

- [x] Mathematical derivation provided (linear wave eigenvector exact for Bx=0 perpendicular case, verified to machine precision)
- [x] Dimensional analysis verified (PF-1000 fill: rho=7.53e-4 kg/m3, p=466 Pa, B=0.01 T, beta=11.7)
- [x] 3+ peer-reviewed citations (Stone et al. ApJS 2008, Miyoshi & Kusano JCP 2005, Borges et al. JCP 2008, Shu & Osher JCP 1988, Scholz Nukleonika 2006)
- [x] Experimental evidence cited (Scholz 2006 I(t) waveform, NRMSE = 0.150 unchanged from prior debates)
- [x] All assumptions explicitly listed (PF-1000 fill conditions at 300K, ideal gas gamma=5/3, Cartesian geometry, smooth initial conditions)
- [x] Uncertainty budget unchanged (5.8% combined Rogowski + digitization)
- [x] All cross-examination criticisms addressed (14 full concessions across 3 panelists, prior eigenvector error claims withdrawn by Dr. DPF and Dr. EE)
- [x] No unresolved logical fallacies
- [ ] Explicit agreement from each panelist — MAJORITY only (2-1: Dr. DPF dissents at 6.4)

### Dissent: Dr. DPF (6.4/10)
Dr. DPF awards +0.1 because: (1) the eigenvector used in Phase AK was proven exact for the perpendicular propagation geometry (Bx=0), which he had previously criticized as ~10% inaccurate — withdrawing this major criticism warrants acknowledgment; (2) the convergence methodology is sound and gamma=5/3 does not affect the measured order. He views these corrections as establishing that Phase AK's verification is of higher quality than he initially assessed. The majority (Dr. PP, Dr. EE) counter that correcting a wrong criticism does not improve the code — it only removes a false negative.

### Path to 7.0 (updated from Debate #22)

| Action | Effort | Projected Delta | Status |
|--------|--------|-----------------|--------|
| Grid convergence study on DPF problem | 2-4 hours | +0.1 | **Done (Phase AK)** — but Cartesian only, not cylindrical |
| R_plasma convergence across 3 resolutions | 4 hours | +0.1 | Partially addressed (snowplow-dominated, not MHD convergence) |
| WENO5 order investigation (transverse resolution) | 2 hours | +0.0 | Identified as unreliable; needs 16+ transverse cells |
| Cylindrical geometry convergence study | 1 day | +0.1 | **Not started** — requires cylindrical solver or coordinate mapping |
| NX2 blind prediction with device-specific pcf | 1 day | +0.2 | Not started |
| MHD spatial validation vs experiment | 2+ days | +0.3 | Not started |
| **Total remaining to 7.0** | | **+0.6** | |

### Debate Statistics
- **Concessions**: 14 full + 0 partial = 14 total
- **Citations retracted**: 0 (prior retractions from Debate #22 stand; Dr. DPF and Dr. EE both withdraw eigenvector error estimates)
- **Error estimates withdrawn**: 2 (Dr. DPF's "~10% eigenvector error", Dr. EE's "4.56% eigenvector error" — both proven wrong for Bx=0 geometry)
- **Score convergence**: 0.1 spread (Dr. PP 6.3, Dr. DPF 6.4, Dr. EE 6.3)
- **7.0 ceiling**: NOT broken (23rd consecutive debate)
- **New findings**: PLM order 1.30 at DPF scales confirmed; WENO5 order 2.14 unreliable (4-cell transverse); float32 degrades order from 1.52 to 1.03; eigenvector exact for perpendicular propagation (consensus)
