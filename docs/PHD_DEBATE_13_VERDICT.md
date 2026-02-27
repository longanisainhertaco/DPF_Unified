# PhD Debate #13 — Verdict

**Date**: 2026-02-27
**Score**: 6.5/10 (up from 6.4)
**Verdict**: MAJORITY (2-1)

## Question
Assess DPF-Unified progress since Debate #12: HLLD momentum fix, _NY_PAD fix, PF-1000 grid fix, Pease-Braginskii diagnostic, bremsstrahlung CGS→SI fix.

## Panel Positions
- **Dr. PP (Pulsed Power)**: AGREE — 6.6 → accepts 6.5
- **Dr. DPF (Plasma Physics)**: AGREE — 6.7 → accepts 6.5
- **Dr. EE (Experimental)**: DISSENT — argues 6.3 (retroactive V&V invalidation)

## Score Breakdown

| Category | Debate #12 | Debate #13 | Delta | Key Factor |
|----------|-----------|-----------|-------|------------|
| MHD | 7.7 | 8.0 | +0.3 | HLLD momentum fix correct (3-0), induction still double-counted |
| Transport | 7.6 | 7.6 | 0.0 | No changes |
| Circuit | 6.7 | 6.7 | 0.0 | No changes |
| DPF | 5.9 | 6.2 | +0.3 | PF-1000 grid fix, PB diagnostic |
| Validation | 5.1 | 5.2 | +0.1 | CT Brio-Wu un-xfailed, offset by _NY_PAD retroactive invalidation |
| AI/ML | 3.5 | 3.5 | 0.0 | No WALRUS progress |
| Software | 7.2 | 7.3 | +0.1 | Grid validation warning, clean conditional HLLD paths |

**Composite**: (8.0 + 7.6 + 6.7 + 6.2 + 5.2 + 3.5 + 7.3) / 7 = **6.50**

## Key Consensus Items (3-0)
1. HLLD momentum flux double-counting fix is mathematically correct (Miyoshi & Kusano 2005)
2. Induction double-counting is architecturally coupled to chain-rule pressure recovery
3. Pease-Braginskii formula correctly implemented per Haines (2011)
4. PF-1000 grid fix necessary but margin thin (0.7 cells beyond cathode)
5. No radiative collapse model despite diagnosing I > I_PB
6. _NY_PAD=4 retroactively invalidates past Python-engine shock tube V&V claims

## Critical Claims Corrected in Cross-Examination
1. **Dr. PP lnΛ direction error**: Called lnΛ=10 "most conservative" — actually LEAST conservative (highest I_PB). Caught by Dr. DPF.
2. **Dr. EE "2.7 cells/pinch"**: Requires 57x compression (unrealistic). Realistic 10-20x gives 7-15 cells. Caught by Dr. DPF.
3. **Dr. DPF R-H error underestimate**: "10-20% for M>3" is actually 26-92% for M=3-10. Caught by Dr. EE.
4. **Dr. PP magnetosonic concern**: Isotropic vs anisotropic error ~4% at PF-1000 beta — quantitatively negligible. Retracted.

## Dissenting Opinion (Dr. EE)
"The _NY_PAD fix reveals that all past Python-engine shock tube V&V was running PLM+Lax-Friedrichs, not WENO5+HLLD as claimed. Bug fixes do not earn score increases — they restore a baseline that should never have been broken. The project has ZERO calibrated experimental validation. Every comparison is code-vs-code or code-vs-analytical. The score should be 6.3, not 6.5."

## New Bugs Identified
- **HLL momentum_flux key bug**: Non-HLLD WENO5 path uses `mass_flux` instead of `momentum_flux` from the HLL solver dict. The conservative `momentum_flux` is computed but unused. Dormant (HLL+WENO5 path adds explicit JxB separately).

## Path to 7.0

| Priority | Action | Impact |
|----------|--------|--------|
| P0 | Evolve E_total conservatively (fix induction + pressure) | +0.3 MHD |
| P0 | Run Athena++ PF-1000 vs Scholz (2006) I(t) data | +0.3-0.5 Validation |
| P1 | Couple bremsstrahlung to MHD energy equation | +0.2 DPF |
| P1 | Increase PF-1000 grid margin (nr=240) | +0.1 DPF |
| P2 | Grid convergence study | +0.1 Validation |

## Citations
1. Miyoshi & Kusano (2005), JCP 208:315-344 — HLLD solver
2. Guo, Xi & Wang (2025), IJNMF 97:1289-1302, DOI:10.1002/fld.5405 — Extended HLLD
3. Haines (2011), PPCF 53:093001 — Pease-Braginskii review
4. Pease (1957), Proc. Phys. Soc. 70:11 — Original PB derivation
5. Stone et al. (2020), ApJS 249:4 — Athena++ methods
6. Scholz et al. (2006), Nukleonika 51(1):79-84 — PF-1000 parameters
