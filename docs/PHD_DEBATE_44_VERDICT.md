# PhD Debate #44 — Phase BH: Cross-Publication Blind Prediction

## VERDICT: CONSENSUS (3-0)

### Question
What is the current PhD-level academic assessment of the DPF-Unified simulator, considering Phase BH: cross-publication blind prediction using genuinely independent Gribkov et al. (2007) 90-point PF-1000 waveform data, with nondimensionalized FIM analysis?

### Answer
Phase BH replaces the retracted reconstructed 16 kV waveform with genuinely independent Gribkov (2007) data -- a real methodological advance. However, the model performs at **parity with naive data transfer** (NRMSE 0.1844 vs baseline 0.1876, margin 1.7% untested statistically). The most important finding is that Scholz and Gribkov waveforms exhibit **qualitatively different plasma dynamics** despite identical operating conditions, implying PF-1000 produces at least two distinct pinch regimes. This limits any deterministic model's predictive capability to the circuit-dominated phase (t < 5 us).

### Score: 6.6/10 (UNCHANGED from Debate #43)

## Sub-Scores

| Category | Weight | Debate #43 | Debate #44 | Delta | Rationale |
|----------|--------|-----------|-----------|-------|-----------|
| MHD Numerics | 0.15 | 8.0 | 8.0 | 0.0 | No changes |
| Transport Physics | 0.10 | 7.5 | 7.5 | 0.0 | No changes |
| Circuit Model | 0.15 | 6.8 | 6.8 | 0.0 | Damped loading ratio corrected (0.476), no code changes |
| DPF-Specific | 0.20 | 5.6 | 5.6 | 0.0 | Cross-pub reveals shot-to-shot limits, not new physics |
| V&V | 0.25 | 5.5 | 5.5 | 0.0 | Independent data acquired (+) but ASME FAIL 3.06 and parity with naive baseline (-) cancel |
| AI/ML | 0.05 | 4.5 | 4.5 | 0.0 | No changes |
| Software Eng. | 0.10 | 7.8 | 7.8 | 0.0 | Gribkov data + FIM nondim are incremental infrastructure |

## Consensus Verification Checklist

- [x] **Mathematical derivation provided** -- NRMSE, ASME E/u_val, FIM nondimensionalization, damped RLC all derived with SI units
- [x] **Dimensional analysis verified** -- All formulas checked: NRMSE [dimensionless], I_RLC [A], T/4 [s], FIM [dimensionless after scaling]
- [x] **3+ peer-reviewed citations** -- Gribkov (2007) DOI:10.1088/0022-3727/40/12/008, Scholz (2006) Nukleonika 51(1):79-84, Lee & Saw (2014) J. Fusion Energy 33:319-335, ASME V&V 20-2009
- [x] **Experimental evidence cited** -- Two independent PF-1000 waveforms (Scholz 26-pt, Gribkov 90-pt)
- [x] **All assumptions explicitly listed** -- 10 assumptions with regime of validity (each panelist)
- [x] **Uncertainty budget** -- u_exp = 6.2-7.9% (chain), u_val = 6.0%, shot-to-shot = 5%, ASME E/u_val = 3.06
- [x] **All cross-examination criticisms addressed** -- 15 criticisms raised, all responded to in Phase 3
- [x] **No unresolved logical fallacies** -- Dr. DPF's 0.2439 retracted, Dr. PP's modus tollens reframed, Dr. EE's false dichotomy resolved with Interpretation C
- [x] **Explicit agreement from each panelist** -- Dr. PP AGREE, Dr. DPF AGREE, Dr. EE AGREE

## Panel Positions

- **Dr. PP (Pulsed Power)**: AGREE -- 6.6/10. Model has no measurable predictive skill above naive data transfer. ASME FAIL 3.06. Infrastructure value acknowledged but does not move score.
- **Dr. DPF (Plasma Physics)**: AGREE -- 6.6/10 (revised from initial 6.7). Retracted 0.2439 and "24% improvement." Qualitatively different pinch dynamics is the key finding.
- **Dr. EE (Electrical Engineering)**: AGREE -- 6.6/10 (revised from initial 6.7). Model at parity with data transfer. 4.2% peak current within Rogowski uncertainty.

## Key Findings

1. **Model at parity with naive data transfer**: NRMSE(model->Gribkov) = 0.1844 vs NRMSE(Scholz->Gribkov) = 0.1876. Margin = 0.003 (1.7%), not statistically tested.

2. **Qualitatively different plasma dynamics**: Scholz 30.5% fractional current range (peak+dip) vs Gribkov 0.29% CV (flat plateau) at 5.7-7.4 us. Factor 38x difference. PF-1000 produces distinct pinch regimes at identical conditions.

3. **ASME E/u_val = 3.06 (FAIL)**: First honest ASME ratio on genuinely independent data.

4. **Peak current error 4.2%**: Within 5% Rogowski uncertainty. Not statistically distinguishable from perfect.

5. **FIM nondimensionalized cond = 6.22e3**: Mathematically correct. Still at boundary-trapped point.

6. **Damped RLC peak = 3.927 MA**: Loading ratio corrected to 0.476.

7. **Gribkov provenance unverified**: RADPF archive source, chain-of-custody not confirmed.

8. **Gribkov argmax ill-defined**: +/-1.44 us from flat plateau.

## Concessions and Retractions (19 total)

### Dr. PP (6: 4 full, 2 partial)
1. FULL: Loading ratio 0.347->0.476 (damped basis)
2. FULL: 4/7 red flags are restatements
3. FULL: Parasitic breakdown withdrawn
4. FULL: Duplicate time points inconsequential
5. PARTIAL: "Zero skill" -- margin is 0.003, not exactly zero
6. PARTIAL: Window choice (t<=12 us) not justified

### Dr. DPF (7: 5 full, 2 partial)
1. **FULL: NRMSE 0.2439 retracted** -- cannot be reproduced
2. **FULL: "24% improvement" retracted** -- correct margin is 1.7%
3. FULL: Segment decomposition retracted
4. FULL: Rise NRMSE mischaracterized
5. FULL: Score increase proposal suspended
6. PARTIAL: 0.59 us timing -- argmax ill-defined
7. PARTIAL: Provenance -- unverified but 3 circumstantial checks

### Dr. EE (6: 4 full, 2 partial)
1. FULL: Point distribution bias (wrong direction)
2. FULL: 10.2% timing jitter retracted
3. FULL: Single-shot u_shot = 5%
4. FULL: Normalization convention aligned
5. PARTIAL: "7-step chain" is 3-component RSS
6. PARTIAL: "Shot-to-shot dominant" revised to "dominant non-canceling"

## Path to 7.0 -- Revised

### HIGHEST PRIORITY
1. **Cross-CONDITION validation** -- calibrate at 27 kV, predict at 16 kV or different device
   - Must beat naive data transfer by > 2 sigma
   - Expected: +0.15-0.25

### HIGH PRIORITY
2. **Verify Gribkov provenance** -- compare RADPF data vs original paper Fig. 4
3. **Multi-shot ensemble calibration** -- average multiple shots, then calibrate
4. **Statistical significance test** -- paired bootstrap on NRMSE margin

### MEDIUM PRIORITY
5. **Characterize Gribkov plateau** -- delayed pinch, weak pinch, or bandwidth artifact?
6. **Add shot-to-shot to ASME u_val** for PF-1000-Gribkov

## Score Progression

| Debate | Phase | Score | Change | Key Finding |
|--------|-------|-------|--------|-------------|
| #36 | X | 6.5 | +0.0 | Baseline |
| #37-42 | AA-BF | 6.5 | +0.0 | 7 debates at 6.5 |
| #43 | BG | 6.6 | +0.1 | blind_predict infrastructure |
| **#44** | **BH** | **6.6** | **+0.0** | **Cross-pub at parity; different pinch regimes** |

---

*Debate #44 conducted 2026-03-02. All 5 phases executed. 19 concessions, 3 major retractions. CONSENSUS 3-0.*
