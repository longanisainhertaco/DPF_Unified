# PhD Debate #12 Verdict: Slow Test Optimization Strategy

**Date**: 2026-02-26
**Question**: Should the team focus on (A) optimizing slow tests for Metal GPU, (B) offloading to WALRUS AI surrogate, or (C) routing through Athena++?
**Score**: 6.4/10 (up from 6.3, reflecting Phase AD engine validation)

## VERDICT: CONSENSUS (3-0) -- Sequenced hybrid strategy led by Athena++

The question as posed is a false trichotomy. All three panelists agree:

1. **Option A (Metal optimization) rejected**: 72/140 slow tests already ARE Metal. Remaining Python tests validate the Python engine specifically. 0.0 PhD impact.
2. **Option B (WALRUS offload) rejected**: WALRUS untrained on DPF data. Circular reasoning to validate a surrogate with itself. Requires Athena++ training data first.
3. **Option C (Athena++) accepted as physics direction**: Only engine with cylindrical + conservative MHD + HLLD + AMR. But requires 1,000-1,500 LOC integration (not 300).

## Correct Strategy (3-0 Consensus)

### Phase 1: Test Hygiene (2-4 hours, 0.0 PhD impact)
- Session-scope Metal and WALRUS fixtures
- Un-mark 6-10 false-slow tests
- Quick CI improvement

### Phase 2: Athena++ DPF Completion (2-4 weeks, +1.0-1.5 PhD impact)
- Fix bremsstrahlung coefficient: 1.69e-32 -> 1.42e-40 (CGS->SI, confirmed 10^8x error)
- Complete circuit coupling via ruser_mesh_data (~300-400 LOC C++)
- Build subprocess orchestration for convergence studies (~500-800 LOC Python)
- Run PF-1000 at 3 resolutions vs Scholz (2006)
- Write 10+ Athena++ DPF validation tests
- Moves Validation from 4.9 to ~6.0-6.5

### Phase 3: WALRUS Fine-Tuning (after Phase 2, +0.5-1.0 on AI/ML)
- Generate 500-2,000 trajectories from validated Athena++ runs
- Fine-tune WALRUS on DPF-specific MHD data
- Note: isotropy mismatch is structural limitation of IsotropicModel
- Only possible after Athena++ validation

## Claims Retracted (7)

| # | Claim | Retracted By | Reason |
|---|-------|-------------|--------|
| 1 | HLLD is "actually HLLC" | Dr. DPF + Dr. EE | Full Miyoshi & Kusano 2005: 4 intermediate states, 5 waves, 6 regions |
| 2 | Parasitic inductance absent | Dr. PP | L0=33.5nH from short-circuit calibration includes all parasitics |
| 3 | R0=2.3mOhm may be DC | Dr. EE | Measured at operating frequency by standard short-circuit fitting |
| 4 | Energy accounting bug | Dr. PP | Documented design decision to avoid double-counting with MHD |
| 5 | WALRUS needs 10^4 trajectories | Dr. DPF | No scaling law; revised to 500-2,000 per published transfer learning |
| 6 | GCI valid for MHD shocks | Dr. EE | Formal order degrades at discontinuities; 1.86 is blended, not formal |
| 7 | 137/140 = verification only | Dr. EE | ~40+ tests validate physics vs analytical solutions |

## Key Corrections

| Finding | Original | Corrected |
|---------|----------|-----------|
| Athena++ integration effort | 300 LOC | 1,000-1,500 LOC (3-0) |
| m=0 growth time | 2 ns | 3.2 +/- 1.6 ns (for a=1mm) |
| Y_bt/Y_th at PF-1000 | 10-100x | 3-10x (near I_PB) |
| Physics validation tests | 1 | ~40+ analytical + ~10 experimental |
| WALRUS training data needed | 10^4 | 500-2,000 trajectories |

## Panel Positions

- **Dr. PP (Pulsed Power)**: AGREE. Conceded 5/6 criticisms. Circuit solver better designed than initially credited. Surviving: no switch model, no insulation coordination for non-crowbarred devices.
- **Dr. DPF (Plasma Physics)**: AGREE. 3 full concessions, 1 retraction (HLLD), 2 partial. Athena++ is only path to physics credibility. Beam-target ratio qualified to 3-10x for PF-1000.
- **Dr. EE (Experimental Validation)**: AGREE. 4 concessions, 2 partial. Project validation posture stronger than initially assessed (~40+ physics tests, not 1). Surviving: Rogowski bandwidth uncharacterized during pinch.

## Subscores (Revised)

| Component | Debate #11 | Debate #12 | Delta | Driver |
|-----------|-----------|-----------|-------|--------|
| MHD Numerics | 7.5 | 7.7 | +0.2 | HLLD confirmed genuine, not HLLC |
| Transport | 7.6 | 7.6 | 0.0 | No change |
| Circuit | 6.6 | 6.7 | +0.1 | Phase AD validates production solver vs experiment |
| DPF-specific | 5.9 | 5.9 | 0.0 | No new DPF physics added |
| Validation | 4.9 | 5.1 | +0.2 | Phase AD (NRMSE=0.1329, 18 tests), reclassified test count |
| AI/ML | 3.5 | 3.5 | 0.0 | No WALRUS progress |
| Software | 7.2 | 7.2 | 0.0 | No change |
| **Composite** | **6.3** | **6.4** | **+0.1** | Phase AD + HLLD confirmation |

## Roadmap to 7.0+

| Priority | Action | Est. Impact | Est. Effort |
|----------|--------|-------------|-------------|
| P0 | Fix bremsstrahlung in dpf_zpinch.cpp | +0.1 (physics correctness) | 1 hour |
| P0 | Complete Athena++ circuit coupling | +0.5-1.0 (validation) | 2-3 weeks |
| P0 | PF-1000 I(t) via Athena++ vs Scholz | +0.5 (experimental validation) | 1 week |
| P1 | WALRUS fine-tuning on Athena++ data | +0.5-1.0 (AI/ML) | 2-4 weeks |
| P1 | Implement Pease-Braginskii check | +0.2 (DPF physics) | 1 week |
| P2 | Cross-device validation (NX2) | +0.3 (validation breadth) | 1 week |

## Key Insights

1. **The slow tests are not the problem -- what they test is.** 72 Metal tests validate Cartesian MHD numerics (irrelevant for DPF). 33 Python tests validate a non-conservative teaching engine. Zero tests validate Athena++ DPF physics.
2. **HLLD is real.** Both Dr. DPF and Dr. EE incorrectly claimed Metal uses HLLC. The code has full 5-wave HLLD with double-star states. This raises MHD numerics subscore.
3. **L0 and R0 are correctly characterized.** Standard DPF short-circuit calibration provides system-level parameters. No parasitic or skin-depth corrections needed.
4. **WALRUS pathway is valid but strictly sequenced.** Train on Athena++ data -> validate on held-out set -> use for parameter sweeps. Cannot shortcut the physics engine.
5. **The project has ~40+ physics validation tests, not 1.** Bennett equilibrium, Rankine-Hugoniot, bremsstrahlung scaling all constitute physics validation per AIAA G-077-1998.
