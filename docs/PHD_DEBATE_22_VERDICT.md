# PhD Debate #22 Verdict: Post-Phase AJ Metal Engine NaN Stability Fix

## VERDICT: CONSENSUS (3-0) — Score: 6.3/10 (unchanged)

### Question
Does Phase AJ (Metal engine NaN instability fix at 64x1x128: positivity-preserving reconstruction fallback + velocity clamping + neighbor-averaging NaN repair) break the 7.0 ceiling?

### Answer
**No.** Phase AJ is competent numerical engineering that follows established best practices (Zhang-Shu positivity limiter, Athena++-style floor enforcement, velocity clamping). It enables the Metal MHD solver to complete PF-1000 simulations at 64x1x128 resolution without NaN crashes. However, it changes zero experimentally validated quantities, adds no new physics, and does not address the validation gaps that define the 7.0 ceiling.

### Panel Positions
- **Dr. PP (Pulsed Power):** 6.3/10 — AGREE. "Software that does not crash is a prerequisite, not an achievement."
- **Dr. DPF (Plasma Physics):** 6.3/10 — AGREE. "The code was broken at 64x1x128; now it runs. This restores functionality that should have existed."
- **Dr. EE (Electrical Engineering):** 6.3/10 — AGREE. "Phase AJ is robustness infrastructure. No new validation data, no convergence study, no uncertainty quantification."

### Phase AJ Technical Assessment

#### Three-Layer Fix
1. **Positivity-preserving reconstruction fallback** (metal_riemann.py): Zhang-Shu-type a priori interface checking. Detects negative pressure, extreme velocity (v^2 > 2.5e11), NaN at reconstructed states. Replaces with first-order donor cell. Follows Stone et al. (ApJS 249, 2020, Sec 4.7).
2. **Velocity clamping** (metal_solver.py): 10x local fast magnetosonic speed after each SSP-RK stage. State-dependent (not hardcoded). SSP guarantee technically unverified for composed operator but empirically supported.
3. **Neighbor-averaging NaN repair** (metal_solver.py): 6-neighbor conv3d kernel replaces NaN/Inf with averaged values. Non-conservative on primitive variables. Never invoked in float32 (0 repairs).

#### Results
- Float32: 0 NaN repairs (was 10,025)
- Float64: Under threshold (was crash)
- 8 previously-xfailed tests now pass
- 2 additional convergence tests XPASS
- All 196 slow tests pass

### Concessions (Phase 3)

**Dr. PP (4 full, 2 partial):**
- MOOD analogy withdrawn (code is Zhang-Shu-type, not MOOD)
- Guo et al. (2025) citation withdrawn as irrelevant
- "500 km/s hardcoded" partially wrong — Euler-stage is 10x c_f (state-dependent)
- "4,945 float64 repairs" withdrawn as unverifiable
- SSP "formally destroyed" softened to "unverified for composed operator"

**Dr. DPF (3 full, 2 partial):**
- 500x under-resolution revised to ~5-15x (d_i not d_e)
- "No instabilities = not physical" withdrawn for Lee model scope
- Balsara & Spicer as methods reference, not DPF validation
- "Spectator" revised to "perturbative during rundown, significant at pinch"

**Dr. EE (2 full, 3 partial):**
- Klower et al. withdrawn as inapplicable to MHD shocks
- 10,000 threshold accepted as reasonable (0.006% repair fraction)
- Verification vs validation: "verification defect detected via validation comparison"
- Float32/float64 concern correctly attributed to Metal solver specifically
- "No convergence study" refined to smooth MHD at DPF-relevant scales

### Sub-Score Impact

| Subsystem | Score | Delta | Rationale |
|-----------|-------|-------|-----------|
| MHD Numerics | 8.0 | 0.0 | Positivity fallback is robustness, not accuracy |
| Transport Physics | 7.5 | 0.0 | No changes |
| Circuit Solver | 6.5 | 0.0 | No changes |
| DPF-Specific Physics | 5.8 | 0.0 | No new DPF physics |
| Validation & V&V | 4.8 | 0.0 | No new validation data |
| AI/ML Infrastructure | 4.0 | 0.0 | No changes |
| Software Engineering | 7.5 | 0.0 | Quality work within existing framework |

### Path to 7.0 (unchanged from Debate #21)

| Action | Effort | Projected Delta |
|--------|--------|----------------|
| Grid convergence study on DPF problem | 2-4 hours | +0.1 |
| R_plasma convergence across 3 resolutions | 4 hours | +0.1 |
| NX2 blind prediction with device-specific pcf | 1 day | +0.2 |
| MHD spatial validation vs experiment | 2+ days | +0.3 |
| **Total ceiling** | | **+0.7 → 7.0** |

### Consensus Verification Checklist

- [x] Mathematical derivation provided (positivity-preserving theory)
- [x] Dimensional analysis verified (c_f, d_i, d_e calculations checked)
- [x] 3+ peer-reviewed citations (Zhang & Shu 2010, Stone et al. 2020, Tann et al. 2019, Klower et al. 2020, Sahyouni et al. 2021, Bellan 2020, Balsara & Spicer 1999)
- [x] Experimental evidence cited (Scholz 2006 I(t) waveform, NRMSE = 0.150 unchanged)
- [x] All assumptions explicitly listed (7 assumptions by Dr. PP, 8 by Dr. DPF, 5 by Dr. EE)
- [x] Uncertainty budget unchanged (5.8% combined Rogowski + digitization)
- [x] All cross-examination criticisms addressed (9 full concessions, 7 partial)
- [x] No unresolved logical fallacies
- [x] Explicit agreement from each panelist (3-0 CONSENSUS at 6.3)

### Debate Statistics
- **Concessions**: 9 full + 7 partial = 16 total
- **Citations retracted**: 2 (MOOD/Tann et al. misapplied, Guo et al. irrelevant)
- **Score convergence**: 0.0 spread (all three at 6.3 from Phase 1)
- **7.0 ceiling**: NOT broken (22nd consecutive debate)
- **New findings**: 0 (Phase AJ assessed as robustness-only)
