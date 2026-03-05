# PhD Debate #50 Verdict — Phase BO: Multi-Condition Validation + MJOLNIR Geometry Fix

## VERDICT: CONSENSUS (3-0) — Score: 6.7/10 (UNCHANGED from 6.7)

### Question
What is the current PhD-level academic assessment of DPF-Unified, considering Phase BO multi-condition validation (same device, different operating conditions), the MJOLNIR anode_radius geometry fix, and the updated N=5 LOO evidence?

### Answer
Phase BO implements multi-condition validation infrastructure and produces three cross-condition transfer results for PF-1000. However, the flagship result (27kV to 16kV, degradation 1.03x) is **circular** because the PF-1000-16kV waveform is reconstructed from the 27kV Scholz training data, not independently measured. The Gribkov result (degradation 1.25x) tests cross-shot reproducibility, not cross-condition transfer, since it uses the same 27kV operating conditions. The MJOLNIR anode_radius fix (76mm to 114mm) corrects a genuine geometry error identified by panel cross-examination in Debate #49, reclassifying MJOLNIR from "super-driven" (S/S_opt=2.81) to "optimal" (S/S_opt=1.04). Both contributions are positive but neither provides the independent experimental evidence needed to break the 7.0 ceiling. Score remains 6.7/10.

### Consensus Verification Checklist

- [x] **Mathematical derivation provided** — Lee model momentum balance, circuit similarity analysis, speed factor S computation
- [x] **Dimensional analysis verified** — V0^2/p0 identified as DIMENSIONAL (not a valid Buckingham Pi group); S = (I_peak/a) / (4pi * sqrt(mu0 * rho0 / (pi * k))) is dimensionless
- [x] **3+ peer-reviewed citations** — Lee & Saw (2008) JPCS, Scholz (2006) Nukleonika, Akel et al. (2021) IEEE TPS, Gribkov et al. (2007) JPhysD, Lee (2005) ICPIG
- [x] **Experimental evidence cited** — PF-1000 Scholz (2006) 27kV I(t), Gribkov et al. (2007) 27kV I(t), Akel et al. (2021) 16kV I(t) (not yet digitized)
- [x] **All assumptions explicitly listed** — 7 assumptions below
- [x] **Uncertainty budget** — Waveform provenance (reconstructed vs digitized), ASME V&V 20 ratios computed
- [x] **All cross-examination criticisms addressed** — circularity, V0^2/p0 dimensional error, circuit theorem, cross-shot vs cross-condition
- [x] **No unresolved logical fallacies** — circular validation identified and documented; no false claims remain
- [x] **Explicit agreement from each panelist** — Dr. PP 6.7, Dr. DPF 6.7, Dr. EE 6.7

### Multi-Condition Validation Results

| Transfer Pair | Train Device | Test Device | Blind NRMSE | Indep NRMSE | Degradation | ASME Ratio |
|--------------|-------------|-------------|-------------|-------------|-------------|------------|
| 27kV to 16kV | PF-1000 (27kV Scholz) | PF-1000-16kV | 0.1187 | 0.1150 | 1.03x | 1.03 FAIL |
| Scholz to Gribkov | PF-1000 (Scholz) | PF-1000-Gribkov | 0.1972 | 0.1575 | 1.25x | 3.27 FAIL |
| Reverse 16kV to 27kV | PF-1000-16kV | PF-1000 (Scholz) | 0.1006 | 0.0963 | 1.04x | 1.48 FAIL |

### MJOLNIR Geometry Fix

| Parameter | Before (Bug) | After (Fixed) | Source |
|-----------|-------------|---------------|--------|
| anode_radius | 76 mm (implosion radius) | 114 mm (physical anode) | cathode 157mm - A-K gap 43mm |
| S/S_opt | 2.81 (super-driven) | 1.04 (optimal) | Lee & Saw speed factor |
| L_p/L0 | 0.37 | 0.16 | Geometry-dependent inductance |

### Supporting Evidence

#### 1. PF-1000-16kV Waveform Is Reconstructed, Not Measured (3-0 Unanimous)

The PF-1000-16kV waveform in `experimental.py` (lines 270-329) is explicitly reconstructed from the 27kV Scholz waveform shape, scaled by voltage ratio. The code comments state: "Replace with actual digitized data from Akel (2021) Fig. 3 when available." This means:

- The 1.03x degradation tests the Lee model's ability to predict a **scaled copy of its own training data**
- The 0.1187 blind NRMSE reflects parameter interpolation error, not physics prediction
- The ASME V&V 20 ratio of 1.03 is an artifact of the circular construction
- **This finding was first identified in Debate #43** and remains the dominant limitation

#### 2. V0^2/p0 Is Dimensional, Not a Valid Similarity Group (3-0 Unanimous)

Both Dr. PP and Dr. DPF initially computed V0^2/p0 as a similarity parameter comparing 27kV/3.5 Torr vs 16kV/5.32 Torr conditions. Cross-examination revealed:

- V0^2/p0 has dimensions of [V^2/Pa] = [m^4 kg / (A^2 s^4)] — not dimensionless
- It cannot be a valid Buckingham Pi group
- The proper Lee model similarity parameter is the **speed factor S** (dimensionless)
- PF-1000 at 27kV: S/S_opt = 0.98 (near-optimal)
- The change from 27kV/3.5 Torr to 16kV/5.32 Torr shifts the operating regime, but V0^2/p0 ratio cannot quantify this properly

#### 3. I(t)/V0 Self-Similarity Is a Circuit Theorem (3-0 Unanimous)

The near-identical I(t)/V0 normalized waveforms between 27kV and 16kV conditions reflect a **circuit property**, not a Lee model physics prediction:

- For a linear RLC circuit: I(t) = (V0/omega*L) * sin(omega*t) * exp(-alpha*t)
- I(t)/V0 eliminates the voltage dependence exactly
- Any model that correctly implements the circuit equations (including bare RLC with no plasma physics) would show this self-similarity
- The self-similarity therefore does **not** validate the snowplow physics (fc, fm)

#### 4. Gribkov 1.25x Is Cross-Shot, Not Cross-Condition (3-0 Unanimous)

The Scholz-to-Gribkov transfer (degradation 1.25x, blind NRMSE 0.1972) tests:

- Same device: PF-1000
- Same operating conditions: 27 kV, ~3.5 Torr D2
- Different publication: Scholz (2006) vs Gribkov et al. (2007)
- Different experimental shot(s)

This is **cross-shot reproducibility**, not cross-condition validation. The 1.25x degradation reflects:
- Shot-to-shot variation in DPF experiments (typically 5-15% for PF-1000)
- Differences in digitization method and density between publications
- Possible differences in gas purity, electrode conditioning, timing trigger

While genuine and informative, it does not test the model's ability to predict across different operating regimes.

#### 5. MJOLNIR Geometry Fix Is Genuine (3-0 Unanimous)

The MJOLNIR anode_radius bug was identified during Debate #49 cross-examination. The fix is well-motivated:

- Original: used implosion radius (76 mm) from `anode_radius` field
- Corrected: physical anode outer radius = cathode_radius (157 mm) - A-K gap (43 mm) = 114 mm
- Impact on speed factor: S/S_opt changes from 2.81 (unrealistic super-driven) to 1.04 (physically reasonable optimal)
- Impact on L_p/L0: changes from 0.37 to 0.16 (even more circuit-dominated)
- The fix was identified by panel process (cross-examination), demonstrating the value of the debate protocol

#### 6. Loading Factor Shift Is Expected Physics (3-0 Unanimous)

The 8.3% shift in loading factor between 27kV (0.59) and 16kV (0.54) conditions is consistent with:

- Higher fill pressure (5.32 vs 3.5 Torr) increases mass loading, reducing sheath velocity
- Lower voltage (16 vs 27 kV) reduces drive current
- Net effect: slight reduction in L_p/L0, consistent with Lee model scaling
- This is a genuine physics prediction of the multi-condition framework, but cannot be validated until real 16kV data is available

#### 7. ASME V&V 20 Remains FAIL for All Conditions (3-0 Unanimous)

All three multi-condition transfer pairs produce ASME FAIL:

- 27kV to 16kV: ratio 1.03 — deceptively close to PASS but circular (reconstructed waveform)
- Scholz to Gribkov: ratio 3.27 — clear FAIL, dominated by cross-shot variation
- Reverse 16kV to 27kV: ratio 1.48 — FAIL
- The original PF-1000 ASME ratio remains 1.540 (FAIL from Debate #49)

#### 8. Phase BO Infrastructure Is Genuine (3-0 Unanimous)

The `multi_condition_validation()` function and `MultiConditionResult` dataclass in calibration.py provide:

- Automated cross-condition calibration and blind prediction
- ASME V&V 20 integration for each transfer pair
- Degradation ratio computation
- 25 non-slow + 9 slow tests in test_phase_bo_multi_condition.py
- Reusable framework ready for real digitized waveform data

### Assumptions and Limitations

1. **Reconstructed waveform circularity**: PF-1000-16kV I(t) is voltage-scaled from 27kV Scholz shape, invalidating any NRMSE-based validation claim for this pair
2. **Same-condition Gribkov**: Tests reproducibility, not cross-condition transfer
3. **MJOLNIR geometry derivation**: Anode radius derived from cathode radius minus A-K gap; no direct measurement of anode radius available from publication
4. **maxiter=3 for calibration**: Quick results but may not reach global optimum; maxiter=10-100 needed for convergence study
5. **Lee model validity**: Assumed valid for both 27kV and 16kV conditions, though fill pressure difference changes sheath dynamics
6. **ASME V&V 20 applicability**: Same-data training and testing violates Section 5.3 for 27kV; cross-condition requires independent data
7. **Speed factor S computation**: Uses published/fitted device parameters; any parameter error propagates to S/S_opt

### Uncertainty

- **Reconstructed waveform uncertainty**: Not quantifiable (systematic error, not statistical)
- **Digitization uncertainty**: 2% for Gribkov (94 points), 5% for 16kV reconstruction
- **ASME V&V 20**: u_val dominated by waveform provenance, not numerical error
- **MJOLNIR S/S_opt**: Uncertainty in A-K gap and cathode radius propagates to ~10% in S

### Panel Positions

- **Dr. PP (Pulsed Power): AGREE — 6.7/10.** The MJOLNIR fix is a genuine correction with physically meaningful impact (super-driven to optimal). The multi-condition infrastructure is well-designed. However, the reconstructed 16kV waveform means no new validation evidence exists until Akel (2021) is digitized.

- **Dr. DPF (Plasma Physics): AGREE — 6.7/10.** The 1.03x degradation would be significant if the waveform were real. The loading factor prediction (0.59 to 0.54) is a genuine physics capability. But circular validation cannot contribute to the score. The V0^2/p0 dimensional error in the initial analyses demonstrates the importance of the cross-examination protocol.

- **Dr. EE (Electrical Engineering): AGREE — 6.7/10.** The I(t)/V0 circuit theorem explains most of the apparent self-similarity. The MJOLNIR geometry fix is properly derived and improves the database. All three ASME ratios FAIL, consistent with the model-form error identified in Debate #36. The infrastructure is publishable; the validation claims are not.

### Dissenting Opinion
None. Unanimous consensus.

### Sub-Scores (unchanged from Debate #49)

| Category | Score | Notes |
|----------|-------|-------|
| MHD Numerics | 8.0 | WENO-Z, HLLD, SSP-RK3, CT (Metal + Python) |
| Transport Physics | 7.5 | Braginskii, Spitzer, GMS Coulomb log |
| Circuit Coupling | 6.8 | RLC + snowplow + sub-cycling + crowbar |
| DPF-Specific | 5.9 | Lee model + 5-device LOO + multi-condition framework |
| V&V | 5.9 | N=5 LOO + ASME + multi-condition (all FAIL) |
| AI/ML | 4.5 | WALRUS integration + surrogate inference |
| Software | 7.8 | 3469 tests, Pydantic config, FastAPI, Metal GPU |

### Recommendations for Further Investigation

1. **CRITICAL: Digitize Akel (2021) Fig. 3** — The 16kV I(t) waveform from Akel et al. (2021) IEEE TPS is the minimum necessary action. Replace the reconstructed waveform to enable genuine cross-condition validation. This is the single highest-impact action for reaching 7.0.

2. **Re-run multi-condition validation with real data** — Once Akel (2021) is digitized, the existing `multi_condition_validation()` infrastructure can immediately produce credible cross-condition results.

3. **Complete N=5 LOO with maxiter >= 5** — Resolve the 3-fold degeneracy identified in Debate #49. Higher maxiter should produce distinct trained parameters for each fold.

4. **Investigate FAETON-I 6.21x degradation** — The extreme outlier in LOO deserves targeted analysis: bare RLC baseline, fm sensitivity, and whether the 100kV operating regime is within Lee model validity.

5. **Stratified LOO reporting** — Report LOO statistics separately for measured (N=3: PF-1000, POSEIDON, UNU-ICTP) vs reconstructed (N=2: FAETON-I, MJOLNIR) waveforms.

### Path to 7.0

| Action | Expected Impact | Status |
|--------|----------------|--------|
| Digitize Akel (2021) 16kV I(t) | +0.15-0.25 | **BLOCKED** (need source) |
| Re-run multi-condition with real data | +0.10-0.15 | Blocked by above |
| N=5 LOO with maxiter >= 5 | +0.02-0.05 | Scripts created, runtime too long |
| ASME PASS on any cross-condition pair | +0.10-0.20 | Requires real data |
| Stratified LOO reporting | +0.02-0.03 | Ready to implement |

**Net potential if Akel (2021) digitized**: +0.25-0.45 (range: 6.95-7.15)

### 7.0 Ceiling Status

**7.0 ceiling NOT broken** (50th consecutive debate).

The ceiling remains primarily due to the absence of independently measured cross-condition waveform data. The infrastructure for multi-condition validation is complete and well-tested. The single blocking item is digitization of real experimental data.
