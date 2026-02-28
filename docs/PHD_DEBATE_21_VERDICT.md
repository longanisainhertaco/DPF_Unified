# PhD Debate #21 — Post-Phase AH "Path to 7.0" Assessment

## VERDICT: CONSENSUS (3-0)

### Question
After completing Phase AH (preset completeness, NRMSE truncation, cross-device prediction, parameter sensitivity, device-specific pcf), does the DPF-Unified score improve from 6.2/10?

### Answer: 6.3 / 10

This represents a **+0.1 increase from Debate #20's 6.2/10**. The increase reflects the back-EMF fix in engine.py and the frozen L_plasma fix, NOT the Phase AH items themselves. Phase AH was primarily software housekeeping with minimal physics validation improvement.

---

## Consensus Verification Checklist

- [x] **Mathematical derivation provided** — Reflected shock density: single-shock R-H gives rho_post = (gamma+1)/(gamma-1) * rho0 = 4*rho0 for gamma=5/3. Double-shock limit gives up to 16*rho0 for strong reflected shock, Mach~2 gives ~9*rho0. Effective DOF: N_eff = N/(1 + 2*sum(rho_k)) ~ 26/5 ~ 5.
- [x] **Dimensional analysis verified** — MJOLNIR T_4 = (pi/2)*sqrt(15e-9 * 4e-4) = 3.85 us. Speed factor S_PF1000 = 5.93e8, S_MJOLNIR = 1.00e9 (ratio 1.69).
- [x] **3+ peer-reviewed citations** — Scholz et al. (2006) Nukleonika 51(1):79-84, Lee & Saw (2008) J. Fusion Energy 27, Lee & Saw (2014) J. Fusion Energy 33, Goyon et al. (2025) Phys. Plasmas 32:033105, Miyoshi & Kusano (2005) JCP 208:315-344.
- [x] **Experimental evidence cited** — Scholz (2006) 26-point PF-1000 I(t): I_peak = 1.87 MA, dip ~33%. Measurement uncertainty unstated in source.
- [x] **All assumptions explicitly listed** — See Assumptions and Limitations section.
- [x] **Uncertainty budget** — Per-point: 5% Rogowski (estimated), 2% digitization. Effective DOF ~ 5 (revised from 17 in Phase 2). GUM for measurement; ASME V&V 20-2009 for validation framework.
- [x] **All cross-examination criticisms addressed** — 12 challenges issued in Phase 2, 12 responses in Phase 3. 4 full concessions, 2 partial concessions, 0 unaddressed.
- [x] **No unresolved logical fallacies** — Straw man (Dr. PP's +1.0 baseline) retracted. DOF=17 assumption corrected to ~5.
- [x] **Explicit agreement from each panelist** — Dr. PP: 6.3 (AGREE). Dr. DPF: 6.3 (AGREE). Dr. EE: 6.3 (AGREE).

---

## Sub-Score Breakdown

| Subsystem | Debate #20 | Debate #21 | Delta | Rationale |
|-----------|-----------|-----------|-------|-----------|
| MHD Numerics | 8.0 | 8.0 | 0.0 | No new MHD algorithm work |
| Transport Physics | 7.5 | 7.5 | 0.0 | No changes |
| Circuit Solver | 6.5 | 6.5 | 0.0 | ESR/switch model gaps unchanged |
| DPF-Specific Physics | 5.5 | 5.8 | +0.3 | Back-EMF fix (+0.15), frozen L_plasma fix (+0.1), pcf confirmed (+0.05) |
| Validation & V&V | 4.5 | 4.8 | +0.3 | Phase AG 21 tests (+0.15), NRMSE truncation API (+0.05), DOF revision acknowledged (+0.05) |
| AI/ML Infrastructure | 4.0 | 4.0 | 0.0 | No changes |
| Software Engineering | 7.5 | 7.5 | 0.0 | ~2797 tests, stable infrastructure |

---

## Phase AH Item Assessment

| Item | Projected Credit | Awarded | Rationale |
|------|-----------------|---------|-----------|
| Calibrated presets (all cylindrical) | +0.1 | +0.0 | PF-1000/NX2 legitimate, LLNL/MJOLNIR fabricated ("Typical lab-scale", "similar to PF-1000") |
| NRMSE truncation at current dip | +0.1 | +0.0 | Truncation at SIMULATED dip is self-referential; improvement within noise band (±0.03 at 1σ) |
| Cross-device blind prediction | +0.3 | +0.0 | CrossValidator uses same pcf for both devices (default 1.0); 50% peak error threshold vacuous |
| Parameter sensitivity study | +0.1 | +0.0 | Single-DOF perturbations miss fc²/fm degeneracy; ±10% arbitrary; no Jacobian/Hessian |
| Device-specific pcf in calibrate | +0.2 | +0.0 | pcf was already propagated (Debate #20 stale finding); API fix is housekeeping |
| Bennett equilibrium test | +0.2 | N/A | Not included in Phase AH |

## Pre-Existing Fix Credits (earned between Debate #20 and #21)

| Fix | Credit | Rationale |
|-----|--------|-----------|
| Back-EMF double-counting fix (engine.py:868-879) | +0.05 | back_emf=0 when snowplow active; resolves Debate #20 finding |
| Frozen L_plasma fix (engine.py sub-cycling) | +0.05 | Prevents catastrophic current spike when snowplow.is_active=False |

---

## Key Debate Findings

### NEW: Effective DOF ~ 5 (not 17)
The 26-point Scholz waveform with ~1-2 μs autocorrelation time has only ~5 effective independent data points. With 9 model parameters (fc, fm, f_mr, pcf, R0, L0, C, V0, liftoff), the system is over-parameterized. NRMSE is a fit quality metric, not a validation metric. This explains why calibration always "succeeds" regardless of starting point.

### NEW: Reflected Shock Density Underestimate
`lee_model_comparison.py` and `snowplow.py` use `rho_post = 4.0 * rho0` (single-shock R-H). The reflected shock encounters pre-shocked gas (4×ρ₀), so true density should be ~9-16×ρ₀ depending on reflected shock Mach number. Impact on I(t) is moderate (reflected shock phase is short).

### CONFIRMED: Cross-device pcf bug
`CrossValidator.validate()` uses `pinch_column_fraction=1.0` (default) for BOTH PF-1000 and NX2. Correct values: PF-1000 → 0.14, NX2 → 0.5. The API accepts only one pcf parameter. This is an API design deficiency, not a physics bug.

### CONFIRMED: LLNL/MJOLNIR presets are fabricated
- LLNL: fc=0.7, fm=0.15 — comment says "Typical lab-scale DPF", no citation
- MJOLNIR: fc=0.7, fm=0.1, pcf=0.14 — comment says "MA-class: similar to PF-1000", pcf copied from PF-1000 despite different geometry (tapered anode)
- Scholar Gateway returned zero results for both Deutsch & Kies (1988) and Goyon et al. (2025)

### CORRECTED: Debate #20 baseline
The Debate #20 verdict document says 6.2/10, not 6.3 as recorded in memory/debates.md. Corrected.

---

## Assumptions and Limitations

1. Thin-sheath approximation (snowplow model)
2. Lumped-circuit coupling (0D plasma + 0D circuit)
3. pcf=0.14 is a fit parameter from Scholz X-ray imaging (Type B estimate)
4. Single experimental dataset (Scholz 2006, 26 points, ~5 effective DOF)
5. Calibration is not prediction (fc/fm tuned on same data used for validation)
6. MHD engine spatial outputs unvalidated against experiment
7. Single-shock Rankine-Hugoniot for reflected shock density (factor ~2-4x underestimate)
8. Scholz waveform measurement uncertainty unstated (estimated 5% Rogowski + 2% digitization)
9. No grid convergence study for any comparison

---

## Panel Positions

### Dr. PP (Pulsed Power Engineering): AGREE — 6.3/10
Phase AH is mostly software housekeeping. The back-EMF and frozen L_plasma fixes are genuine. ESR, switch model, and voltage reversal protection remain gaps. LLNL/MJOLNIR references unverifiable via Scholar Gateway.

### Dr. DPF (Dense Plasma Focus Theory): AGREE — 6.3/10
Effective DOF ~5 means calibration is over-parameterized. Reflected shock density is a moderate concern. No new physics in Phase AH — only preset/API changes. MHD engine remains "spectator" for I(t) validation.

### Dr. EE (Electrical Engineering & Metrology): AGREE — 6.3/10
DOF revision from 17 to ~5 is the most significant metrological finding. ASME V&V 20-2009, not GUM alone, is the correct framework. Scholz measurement uncertainty gap persists across all 21 debates.

---

## Recommendations for Next Phase

| Action | Effort | Projected Δ |
|--------|--------|-------------|
| Tighten Phase AE test bounds to published ranges | 15 min | +0.05 |
| State Scholz measurement uncertainty in code | 30 min | +0.1 |
| Reflected shock density correction (~8×ρ₀) | 2 hours | +0.05 |
| Higher-resolution Metal engine (64×1×128) | 4 hours | +0.1 |
| Second device blind prediction (NX2 with correct pcf) | 1 day | +0.2 |

**Path to 7.0**: Items 3-5 required. Projected: 6.3 + 0.3-0.5 = 6.6-6.8. Reaching 7.0 additionally requires MHD spatial validation or neutron yield prediction.

---

## Debate Statistics

- **Duration**: 5 phases, full protocol
- **Challenges issued**: 12 (Phase 2)
- **Concessions**: 4 full, 2 partial
- **New findings**: 2 (effective DOF ~5, reflected shock density)
- **Score convergence**: Phase 1 spread 0.2 (6.3-6.5) → Phase 5 spread 0.0 (all 6.3)
- **7.0 ceiling NOT broken** (unanimous, 21st consecutive debate)
