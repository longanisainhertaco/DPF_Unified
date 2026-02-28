# PhD Debate #24 Verdict: Post-Phase AL Shock Convergence at DPF Conditions

## VERDICT: CONSENSUS (3-0) — Score: 6.3/10 (unchanged)

### Question
Does Phase AL (shock convergence at PF-1000 fill conditions: Sod exact solution comparison, Brio-Wu self-convergence, repair fraction under shocks) break the 7.0 ceiling?

### Answer
**No.** Phase AL is competent verification that confirms expected textbook results for shock convergence. PLM+HLL order 0.85 (Sod) and 0.77 (Brio-Wu) are consistent with theory. WENO5 advantage collapses from 8.2x (smooth) to 1.3x (shocks). Float32 = float64 for shock-dominated flows. All three panelists agree: the project is caught in a verification loop. Four consecutive debates (21-24) have assessed solver verification infrastructure without producing a single new experimental validation data point. The path to 7.0 requires validation against experimental measurements, not more convergence studies.

### Panel Positions
- **Dr. PP (Pulsed Power):** 6.3/10 — "Show me the current waveform match, or the convergence rate is just a number. The code gets shock tube convergence right but cannot predict device current within 10%. That is a verified numerics library, not a validated DPF simulator."
- **Dr. DPF (Plasma Physics):** 6.3/10 — "This is the fourth consecutive debate on Cartesian solver verification. Every MHD code passes Sod and Brio-Wu. The project needs cylindrical geometry, J x B source terms, and radiation-MHD coupling — not more 1D shock tubes."
- **Dr. EE (Electrical Engineering):** 6.3/10 — "The float32 finding is practically useful. The exact Riemann solver is correctly implemented. But none of this moves the project closer to experimental validation. The gap between convergence studies and calibrated measurements remains as wide as in Debate #23."

### Key Findings

1. **Sod PLM+HLL**: L1 order 0.85 (consistent with theory, contact + shock composite)
2. **Brio-Wu PLM+HLL**: self-convergence order 0.77
3. **Brio-Wu WENO5+HLLD**: self-convergence order 0.49 (artifact of sharper resolution in self-convergence metric — see Dr. EE analysis)
4. **WENO5 advantage collapse**: 8.2x (smooth) to 1.3x (shocks) — PLM+HLL nearly as good as WENO5+HLLD for shock-dominated DPF flows
5. **Float32 = Float64**: identical L1 errors at nx=128 — truncation error dominates by ~160x over round-off
6. **Gamma inconsistency**: Brio-Wu uses gamma=2.0 (benchmark convention) while Sod uses gamma=5/3 (monatomic)

### Critical Diagnosis: Verification Loop

The panel unanimously identifies a **verification loop** pattern:
- Phase AH: software housekeeping (Debate #21)
- Phase AJ: Metal NaN fix / robustness (Debate #22)
- Phase AK: smooth wave convergence (Debate #23)
- Phase AL: shock convergence (Debate #24)

**Four consecutive phases of solver infrastructure with zero experimental validation advances.**

### What Would Break 7.0 (Panel Consensus)

| Action | Score Impact | Effort | Priority |
|--------|-------------|--------|----------|
| Improve Metal engine I(t) NRMSE (0.31 to <0.15) | +0.3-0.5 | Medium | **HIGHEST** |
| I(t) with uncertainty budget (Rogowski + digitization) | +0.2 | Low | HIGH |
| Cross-device NX2 I(t) with NRMSE < 0.20 | +0.2 | Medium | HIGH |
| Cross-code comparison vs Athena++ (identical problem) | +0.1 | Low | MEDIUM |
| MHD spatial validation (sheath position, pinch radius) | +0.3 | High | FUTURE |

### Sub-Score Impact

| Subsystem | Score | Delta | Rationale |
|-----------|-------|-------|-----------|
| MHD Numerics | 8.0 | 0.0 | Already 8.0; shock tests confirm, do not advance |
| Transport Physics | 7.5 | 0.0 | No changes |
| Circuit Solver | 6.5 | 0.0 | No changes |
| DPF-Specific Physics | 5.8 | 0.0 | No cylindrical, no J x B, no radiation |
| Validation & V&V | 4.8 | 0.0 | No experimental comparison |
| AI/ML Infrastructure | 4.0 | 0.0 | No changes |
| Software Engineering | 7.5 | 0.0 | More tests, diminishing returns |

### Consensus Verification Checklist

- [x] Mathematical derivation provided (exact Riemann solver, convergence theory)
- [x] Dimensional analysis verified (PF-1000 fill: rho=7.53e-4, p=466 Pa)
- [x] 3+ peer-reviewed citations (Toro 2009, Brio & Wu 1988, Borges 2008, LeVeque 2002, Miyoshi & Kusano 2005)
- [x] Experimental evidence cited (Scholz 2006, unchanged)
- [x] All assumptions explicitly listed
- [x] Uncertainty budget unchanged (5.8%)
- [x] All criticisms addressed (gamma inconsistency noted, self-convergence metric limitation identified)
- [x] No unresolved logical fallacies
- [x] Explicit agreement from each panelist (3-0 CONSENSUS at 6.3)

### Debate Statistics
- **Concessions**: 0 (Phase 1 only — abbreviated due to unanimous agreement)
- **Score convergence**: 0.0 spread (all three at 6.3)
- **7.0 ceiling**: NOT broken (24th consecutive debate)
- **Format**: Abbreviated — Phase 1 only (unanimous agreement, no cross-examination needed)
- **New diagnosis**: "Verification loop" identified by all three panelists independently
