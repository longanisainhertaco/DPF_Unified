# PhD Debate #48 Verdict — Phase BL.1: UNU-ICTP Third Device + N=3 LOO

## VERDICT: CONSENSUS (3-0) — Score: 6.6/10 (UNCHANGED from Debate #47)

### Question
What is the current PhD-level academic assessment of DPF-Unified, considering Phase BL.1: UNU-ICTP third device integration with N=3 leave-one-out cross-validation?

### Answer
Phase BL.1 adds genuine calibration infrastructure (ODE safety limits, device-specific defaults, bounded polish, third device waveform) but does **not** advance the validation status of the plasma physics model. The UNU-ICTP PFF is circuit-dominated (L_p/L0 = 0.35), meaning its I(t) waveform is primarily shaped by the RLC circuit, not the snowplow plasma dynamics. The N=3 LOO upgrades the sampling distribution from Cauchy (df=1, undefined mean) to t(df=2, finite mean but infinite variance), which is a genuine statistical improvement but yields a non-informative 95% CI of [-0.047, 0.345]. The score remains 6.6/10.

### Consensus Verification Checklist

- [x] **Mathematical derivation provided** — L_p/L0 analysis, LOO statistics, RESF computation, bare RLC baseline
- [x] **Dimensional analysis verified** — All formulas checked: RESF [dimensionless], L_p [H], S [kA/(cm·sqrt(Torr))], t-variance [∞ at df=2]
- [x] **3+ peer-reviewed citations** — Lee et al. (1988) Am. J. Phys., Lee & Saw (2014) J. Fusion Energy, Sahyouni et al. (2021) Adv. HEP, ASME V&V 20-2009
- [x] **Experimental evidence cited** — UNU-ICTP waveform (45 points, IPFS archive), PF-1000 Scholz (2006), POSEIDON IPFS
- [x] **All assumptions explicitly listed** — 8 assumptions per panelist with regime of validity
- [x] **Uncertainty budget** — LOO CI width, fc/fm propagated uncertainty, u_quant corrections
- [x] **All cross-examination criticisms addressed** — 100% response rate from all panelists (see concession tallies)
- [x] **No unresolved logical fallacies** — All identified fallacies (inconsistent premises, equivocation, ground truth conflation) were conceded and corrected
- [x] **Explicit agreement from each panelist** — All three independently converge on 6.6/10

### Supporting Evidence

#### 1. UNU-ICTP Is Circuit-Dominated (3-0 Unanimous)
- L_p/L0 = 38.86 nH / 110 nH = 0.353
- Bare RLC (undamped) gives 31.9% peak current error; damped RLC gives 13.9%
- Lee model NRMSE = 0.054 (5.4%) — improvement comes from fitting the damping, not plasma physics
- ASME PASS for UNU-ICTP is non-discriminating: u_exp = 0.117 >> NRMSE = 0.054

#### 2. N=3 LOO: Statistically Valid but Practically Uninformative (3-0 Unanimous)
- t-distribution df=2: finite mean (improvement over df=1 Cauchy), but infinite variance
- 95% CI = [-0.047, 0.345] — includes zero, width = 263% of mean
- LOO degradation: PF-1000 1.66x, POSEIDON 3.72x, UNU-ICTP 1.22x
- Low UNU-ICTP degradation (1.22x) reflects circuit dominance, not shared-parameter success

#### 3. POSEIDON 3.72x Degradation Is the Dominant Signal (3-0 Unanimous)
- Shared NRMSE 0.223 vs independent 0.060 for POSEIDON
- Demonstrates Lee model parameters are device-specific, not universal
- fc²/fm varies 3-fold across devices: 0.87 (POSEIDON), 2.55 (UNU-ICTP), 5.00 (PF-1000)

#### 4. Infrastructure Improvements Are Real (3-0 Unanimous)
- ODE safety limit (500k RHS evals) prevents solve_ivp hangs — genuine robustness
- Device-specific defaults (_DEFAULT_DEVICE_PCF, _DEFAULT_CROWBAR_R) — good engineering
- Bounded L-BFGS-B polish (maxiter=50) replaces unbounded scipy polish
- 3370 non-slow tests passing

### Bugs/Inconsistencies Discovered During Debate

| Issue | Severity | Status |
|-------|----------|--------|
| pcf discrepancy: presets.py 0.06 vs calibration.py 0.50 for UNU-ICTP | HIGH | OPEN |
| Crowbar inconsistency: presets.py crowbar_enabled=True vs calibration crowbar_R=0 | MEDIUM | OPEN |
| V0 discrepancy: presets.py 14.0 kV vs experimental.py 13.5 kV for UNU-ICTP | LOW | OPEN (documented) |
| u_quant in codebase uses 6% (full step/peak) vs GUM-correct 1.6% | MEDIUM | OPEN |

### Major Retractions During Debate

| Panelist | Claim Retracted | Reason |
|----------|----------------|--------|
| Dr. PP | fc²/fm "2.4x variation" | Cannot produce the two calibrations; within-PF-1000 is 1.61x |
| Dr. PP | RESF = 0.198 "high" | PF-1000 RESF = 0.459 is 2.3x higher, not flagged |
| Dr. PP | Score 6.7 | Sub-score arithmetic gives 6.56, rounds to 6.6 |
| Dr. DPF | (L_p/L0)² information proxy | Not derived from information theory; sensitivity ~ L_p/L_total (linear, not quadratic) |
| Dr. DPF | "34 devices needed" for power | Inconsistent with infinite-variance claim; withdrawn |
| Dr. DPF | Sensitivity scales as (L_p/L0)² | Fisher information ≠ sensitivity; ratio is 2-4x not 11x |
| Dr. EE | u_quant = 2.7% | GUM-correct is 1.59%; used maximum error not standard uncertainty |
| Dr. EE | Bare RLC error = 13.4% | Used damped formula (13.9%) but labeled "undamped"; true undamped is 31.9% |
| Dr. EE | Statistical power = 44% | No effect size specified; withdrawn |

### Concession Tallies

| Panelist | Full Concessions | Partial Concessions | Defenses | Retractions |
|----------|-----------------|--------------------| ---------|-------------|
| Dr. PP | 9 | 2 | 1 | 3 |
| Dr. DPF | 5 | 7 | 2 | 4 |
| Dr. EE | 6 | 3 | 0 | 3 |
| **Total** | **20** | **12** | **3** | **10** |

### Sub-Scores

| Category | Weight | Debate #47 | Debate #48 | Delta | Rationale |
|----------|--------|-----------|------------|-------|-----------|
| MHD Numerics | 0.15 | 8.0 | 8.0 | 0.0 | No MHD solver changes |
| Transport Physics | 0.10 | 7.5 | 7.5 | 0.0 | No transport changes |
| Circuit Model | 0.15 | 6.8 | 6.8 | 0.0 | UNU-ICTP tests circuit, not new physics; crowbar inconsistency found |
| DPF-Specific | 0.20 | 5.8 | 5.8 | 0.0 | UNU-ICTP circuit-dominated; pcf 8.3x discrepancy found |
| V&V | 0.25 | 5.7 | 5.7 | 0.0 | LOO df=2 is valid but CI non-informative; ASME PASS non-discriminating |
| AI/ML | 0.05 | 4.5 | 4.5 | 0.0 | No AI changes |
| Software Eng. | 0.10 | 7.8 | 7.8 | 0.0 | ODE safety limit + device defaults are good but offset by pcf/crowbar bugs |

**Weighted sum: 6.56 → rounds to 6.6/10**

### Panel Positions
- **Dr. PP (Pulsed Power)**: AGREE 6.6/10 — "Infrastructure is improving but UNU-ICTP validates the circuit, not the plasma. Two real code bugs discovered (pcf, crowbar). N >= 4 needed for finite sampling variance."
- **Dr. DPF (Plasma Physics)**: AGREE 6.6/10 — "UNU-ICTP information content for constraining fc/fm is 2-4x lower than PF-1000 (linear L_p/L_total scaling). Three major retractions strengthen the assessment framework. Score unchanged."
- **Dr. EE (Electrical Engineering)**: AGREE 6.6/10 — "ASME V&V 20 is non-discriminating for L_p/L0 < 0.5 devices. Codebase u_quant inflated 3.8x above GUM. LOO CI at df=2 has infinite variance. Corrections improve rigor without changing conclusion."

### Key Findings (Hardened by Cross-Examination)

1. **F1 (3-0)**: UNU-ICTP with L_p/L0 = 0.35 is circuit-dominated. Its ASME PASS is non-discriminating — any model with NRMSE < 12% would pass.
2. **F2 (3-0)**: N=3 LOO with t(df=2) has finite mean but infinite variance. The 95% CI is [-0.047, 0.345], which is non-informative. N >= 4 required for finite sampling variance.
3. **F3 (3-0)**: Sensitivity of I(t) to fc/fm scales as L_p/L_total (linear), NOT (L_p/L0)² (quadratic). The (L_p/L0)² proxy is retracted.
4. **F4 (3-0)**: pcf discrepancy 8.3x (presets.py: 0.06 vs calibration.py: 0.50) is a real code inconsistency.
5. **F5 (3-0)**: Crowbar inconsistency (presets.py: enabled, calibration: disabled) affects post-peak I(t) shape.
6. **F6 (3-0)**: IPFS fc/fm values are Lee-model outputs, not experimental measurements. Code-vs-code comparisons should not be labeled "discrepancies from published values."
7. **F7 (3-0)**: LOO degradation values (1.22x, 1.66x, 3.72x) are descriptively informative as individual blind predictions, but statistically uninformative as population estimates.
8. **F8 (3-0)**: u_quant in codebase (6%) is 3.8x above GUM-correct value (1.59%). Should be corrected.
9. **F9 (3-0)**: PF-1000 RESF = 0.459 is higher than UNU-ICTP RESF = 0.198 — both moderate-to-high.
10. **F10 (3-0)**: ASME V&V 20 needs redesign for circuit-dominated devices: either use non-I(t) observables or restrict comparison to the current-dip window.

### Path to 7.0 (Updated)

| Route | Action | Expected Delta | Status |
|-------|--------|---------------|--------|
| A1 | 4th device with L_p/L0 > 1 + digitized waveform | +0.10-0.15 | BLOCKED: no waveform for SPEED-2, KPF-4 |
| A2 | PF-1000 at different V0 (genuinely blind) | +0.15-0.25 | BLOCKED: Akel (2021) is reconstructed from training data |
| A3 | Non-I(t) observable (sheath velocity, neutron timing) | +0.10-0.20 | BLOCKED: requires experimental collaboration |
| B1 | Fix pcf inconsistency (presets.py vs calibration.py) | +0.02 | ACHIEVABLE (bookkeeping fix) |
| B2 | Fix crowbar inconsistency | +0.01 | ACHIEVABLE (code fix) |
| B3 | Fix u_quant to GUM-correct 1.59% | +0.01 | ACHIEVABLE (1-line fix) |
| C1 | N >= 6 devices for meaningful LOO | +0.10-0.20 | BLOCKED: need 3 more digitized waveforms |

**7.0 ceiling remains unbroken (48th consecutive debate).**

### Recommendations for Further Investigation

1. **Fix pcf inconsistency**: Determine whether UNU-ICTP pcf should be 0.06 (presets.py) or 0.5 (calibration.py) and make consistent. Consult Lee & Saw (2009) for small-device pcf values.
2. **Fix crowbar inconsistency**: Set UNU-ICTP preset crowbar_enabled = False (or add crowbar_resistance=0 explicitly) to match calibration behavior.
3. **Correct u_quant**: Change `waveform_digitization_uncertainty=0.06` to `0.016` in experimental.py for UNU-ICTP (GUM-correct value).
4. **Seek 4th device with L_p/L0 > 1**: SPEED-2, KPF-4, or PF-1000 at non-27kV condition would provide the most valuable validation increment.
5. **Consider information-weighted LOO**: Weight devices by L_p/L_total in the LOO framework to emphasize plasma-significant devices.
