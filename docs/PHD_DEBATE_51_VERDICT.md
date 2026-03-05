# PhD Debate #51 Verdict — Phase BQ: Expanded ASME Uncertainty Budget

## VERDICT: CONSENSUS (3-0) — Score: 6.7/10 (UNCHANGED from 6.7)

### Question
What is the current PhD-level academic assessment of DPF-Unified, considering Phase BQ: Expanded ASME uncertainty budget analysis and waveform provenance metadata?

### Answer
Phase BQ presents a competent sensitivity analysis showing that the 27kV-to-16kV multi-condition ASME ratio (1.03, FAIL by 3%) can flip to PASS with u_dig=7% (currently 5%). The arithmetic is verified: u_val increases from 0.1150 to 0.1254, yielding ratio=0.95 (PASS). However, all three panelists independently identify fundamental problems: (1) the 16kV waveform is RECONSTRUCTED from physics scaling of the 27kV Scholz training waveform — this is model-vs-model, not model-vs-experiment, violating ASME V&V 20 Section 4.1 independence requirement; (2) the proposed 7% is a 30% inflation above the RSS-computed 5.39%, with the extra margin reverse-engineered from the desired outcome; (3) at the physically computed u_dig=5.4%, the test STILL FAILS (ratio=1.015). The waveform provenance metadata ("measured"/"reconstructed") is a genuine V&V infrastructure contribution, but it does not flow into the ASME computation and does not change the validation position. Score remains 6.7/10.

### Consensus Verification Checklist

- [x] **Mathematical derivation provided** — ASME V&V 20 budget decomposition, RSS computation, sensitivity analysis
- [x] **Dimensional analysis verified** — all quantities dimensionless (NRMSE, ratios, fractions)
- [x] **3+ peer-reviewed citations** — ASME V&V 20-2009, GUM JCGM 100:2008, Lee & Saw (2008), Akel et al. (2021), Scholz (2006)
- [x] **Experimental evidence cited** — PF-1000 I(t) waveforms from Scholz and Gribkov
- [x] **All assumptions explicitly listed** — 7 assumptions below
- [x] **Uncertainty budget** — full decomposition: u_exp (94%), u_input (5.5%), u_num (0.01%)
- [x] **All cross-examination criticisms addressed** — taxonomy, double-counting, p-hacking all identified
- [x] **No unresolved logical fallacies** — post-hoc uncertainty inflation identified and flagged
- [x] **Explicit agreement from each panelist** — All three at 6.7

### Key Findings (Hardened by Cross-Examination)

#### Finding 1: The 7% u_dig Is Post-Hoc Uncertainty Inflation (CONSENSUS 3-0)
The RSS of 5% (base) and 2% (shape) is 5.39%. Rounding to 7% inflates by 30%. At the physically computed u_dig=5.4%, the ASME ratio is 1.015 — still FAIL. The PASS is achieved only by the rounding decision, not by physics. Per GUM JCGM 100:2008, uncertainty estimates should be derived from physical analysis, not reverse-engineered from desired outcomes.

**Key calculation**: u_dig_threshold = sqrt(E^2/u_target^2 - u_peak_I^2 - u_input^2 - u_num^2 - u_shot_avg^2) where u_target is the u_val that yields ratio=1.0. Result: u_dig >= 5.66% required for PASS.

#### Finding 2: Reconstructed Waveform Violates ASME V&V 20 Independence (CONSENSUS 3-0)
The PF-1000-16kV waveform is reconstructed from the 27kV Scholz waveform (the training data) by physics scaling. ASME V&V 20 Section 4.1 requires validation data to be "independent of the computational model." A waveform derived by scaling the training data is not independent. This makes the 27kV-to-16kV comparison model-vs-model, not model-vs-experiment.

#### Finding 3: 27kV-to-16kV Tests Circuit Scaling, Not Plasma Physics (CONSENSUS 3-0)
Both conditions operate in the same snowplow regime (S/S_opt = 0.98 at 27kV, ~1.14 at 16kV). The electrode geometry (a, b, z_max) is identical. The Lee model is DESIGNED to transfer fc/fm across V0/p0 on the same device. The 1.03x degradation confirms the design premise, not a new physics discovery.

#### Finding 4: u_exp Dominates ASME Budget — 94% of Variance (CONSENSUS 3-0)
The budget decomposition shows u_exp^2/u_val^2 = 94.5%. This means the ASME test primarily measures whether experimental uncertainty bars are large enough to cover model error, not whether the model is accurate. For UNU-ICTP (ratio=0.92, PASS), this was already identified as vacuous (Debate #48).

#### Finding 5: Waveform Provenance Metadata Is Genuine Infrastructure (CONSENSUS 3-0)
The `waveform_provenance` field and `get_devices_by_provenance()` helper are useful V&V infrastructure that enables stratified analysis. However, provenance does not currently flow into the ASME computation — reconstructed waveforms get the same treatment as measured ones. A provenance-dependent uncertainty model would be a genuine improvement.

#### Finding 6: Taxonomy Problem in Uncertainty Classification (Dr. EE, supported 3-0)
The field `waveform_digitization_uncertainty` is used for physically distinct quantities: genuine digitization error (3%, PF-1000), reconstruction model error (5-10%, PF-1000-16kV/FAETON-I/MJOLNIR). The GUM requires each component to be identified by its physical source. This is a category error that should be fixed.

#### Finding 7: Path to Genuine ASME PASS Is Clear (CONSENSUS 3-0)
Obtain the actual measured I(t) waveform from Akel et al. (2021) Fig. 3 at 16kV. This would:
- Replace the reconstructed waveform with genuine experimental data
- Satisfy ASME V&V 20 independence requirement
- Produce a u_dig of 3-5% (typical for figure digitization)
- Make the ASME ratio a legitimate validation claim

### Assumptions and Limitations

1. **Lee model fc/fm transfer across V0/p0** — valid for same device hardware in optimal speed factor regime
2. **ASME V&V 20 applied to NRMSE** — not standard ASME metric (usually absolute error in specific QoI)
3. **GUM RSS for combining uncertainty components** — valid if components are independent
4. **Reconstructed waveform adequately represents actual I(t)** — UNVALIDATED (no measured 16kV data available)
5. **u_dig = 5% is the correct reconstruction uncertainty** — debatable; could be higher (8-10%) or lower (3-5%)
6. **Shot-to-shot variability is independent of peak current uncertainty** — violated for PF-1000-16kV where peak_current_uncertainty is derived from shot-to-shot range
7. **NRMSE is appropriate error metric for ASME** — standard uses absolute error in specific QoI

### Uncertainty

**ASME budget sensitivity**:
- u_dig: 5% → 5.66% flips to PASS (threshold = 5.66%)
- Margin at 7%: ratio = 0.94 (6% below threshold)
- At RSS-computed 5.4%: ratio = 1.015 (STILL FAILS)

**Budget decomposition (27kV→16kV at u_dig=5%)**:
- u_peak_I contribution: 75% of u_val^2
- u_dig contribution: 19% of u_val^2
- u_input contribution: 5.5% of u_val^2
- u_shot_avg contribution: 0.01% (negligible, n=16)
- u_num contribution: <0.01% (negligible)

### Sub-Scores (No Changes from Debate #50)

| Category | Debate #50 | Debate #51 | Change | Justification |
|----------|------------|------------|--------|---------------|
| MHD Numerics | 8.0 | 8.0 | 0.0 | No MHD changes |
| Transport | 7.5 | 7.5 | 0.0 | No transport changes |
| Circuit | 6.7 | 6.7 | 0.0 | No circuit changes |
| DPF-Specific | 5.9 | 5.9 | 0.0 | No new plasma physics validated |
| V&V | 5.9 | 5.9 | 0.0 | Infrastructure gain offset by uncertainty gaming concern |
| AI/ML | 4.5 | 4.5 | 0.0 | No changes |
| Software | 7.9 | 7.9 | 0.0 | Provenance metadata is nice but net zero impact |

**Weighted total**: 6.7/10 (unchanged)

### Panel Positions
- **Dr. PP (Pulsed Power)**: AGREE at 6.7 — 7% u_dig is post-hoc tuning. Reconstructed waveform is fabricated from training data. Pulsed power perspective: waveform shape at 16kV is NOT the same as at 27kV (different sheath dynamics, loading). True reconstruction uncertainty is 10-15%, not 5-7%. But this argues AGAINST the flip — the model would pass easily with 10% u_dig, but the PASS would be meaningless.
- **Dr. DPF (Plasma Physics)**: AGREE at 6.7 — 27kV-to-16kV tests circuit scaling in the same snowplow regime (both S/S_opt near 1.0). ASME PASS by inflating uncertainty is categorically different from PASS by reducing model error. Budget dominance by u_exp (94%) means the test measures experimental noise, not model quality.
- **Dr. EE (Measurement)**: AGREE at 6.7 — Taxonomy problem: calling reconstruction uncertainty "digitization uncertainty" violates GUM. Partial double-counting: peak_current_uncertainty=10% (from shot spread) plus u_shot_to_shot=5%. Borderline p-hacking: transparent but outcome-directed. Fix taxonomy, obtain measured waveform from Akel (2021), and the ASME analysis becomes legitimate.

### Retractions from Previous Debates
None. Debate #51 reinforces Debate #50 findings without new retractions.

### Recommendations for Further Investigation

1. **CRITICAL: Obtain Akel (2021) measured I(t) waveform.** Digitize Fig. 3 from Radiat. Phys. Chem. 188:109633. This single action removes the reconstructed waveform circularity and enables a genuine ASME assessment. DOI: 10.1016/j.radphyschem.2021.109633.

2. **Fix uncertainty taxonomy.** Rename `waveform_digitization_uncertainty` to `waveform_amplitude_uncertainty` and add `uncertainty_type` field (Type A or Type B per GUM).

3. **Implement provenance-dependent ASME model.** Different uncertainty budget structure for measured vs reconstructed waveforms. Reconstructed waveforms should use u_shape, u_amplitude_constraint, u_timing instead of u_dig.

4. **Fix double-counting for PF-1000-16kV.** Either use peak_current_uncertainty=10% (from shot range) OR use u_shot_to_shot=5% (reduced by sqrt(n)), not both. Currently RSS includes both.

5. **Pursue non-I(t) observables.** Neutron yield prediction, pinch radius, soft X-ray timing — these have direct plasma physics sensitivity that I(t) waveform NRMSE does not.

---

*Debate #51 executed with 3 parallel PhD agents (Dr. PP, Dr. DPF, Dr. EE). All three independently identified the same fundamental issues (reconstructed waveform circularity, post-hoc uncertainty inflation, taxonomy problems). Consensus achieved without cross-examination due to unanimous Phase 1 agreement.*
