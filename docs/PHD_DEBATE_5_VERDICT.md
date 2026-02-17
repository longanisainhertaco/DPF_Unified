# PhD Debate Assessment #5: Phases T-U (Snowplow + Cylindrical Geometry)

**Date**: 2026-02-17
**Scope**: Phases T (Snowplow Coupling) and U (Metal Cylindrical Geometry)
**Panel**: Dr. PP (Pulsed Power), Dr. DPF (Plasma Physics), Dr. EE (Electrical Engineering)
**Protocol**: 5-phase debate (Independent Analysis → Cross-Examination → Rebuttal → Synthesis → Verdict)
**Moderator**: Claude Opus 4.6

---

## EXECUTIVE SUMMARY

**VERDICT: CONSENSUS (3-0)**

**Composite Score: 5.6/10** (up from 5.2/10 in Debate #4)

**Characterization**: The simulator remains **bifurcated** — strong generic MHD infrastructure (7.5+) but weak DPF-specific physics (4.8) and validation (2.7). Phases T and U represent the **first qualitative leap in DPF physics capability** since project inception, with the snowplow model producing the current dip signature for the first time. However, validation methodology remains fundamentally flawed, and several critical physics components are missing or incorrect.

**Key Achievement**: Current dip produced (59% depth) — first time ever for this simulator.

**Critical Gaps**: Validation uses wrong peak metric, radial phase missing adiabatic back-pressure (causing 59% dip vs 20-40% experimental), no insulator phase, no parameter calibration framework.

---

## SCORE BREAKDOWN

### Composite: 5.6/10 (up from 5.2/10)

| Component | Debate #4 | Debate #5 | Change | Weight | Contribution |
|-----------|-----------|-----------|--------|--------|--------------|
| **MHD Numerics** | 7.5/10 | 7.75/10 | +0.25 | 20% | 1.55 |
| **Transport** | 7.5/10 | 7.5/10 | 0 | 15% | 1.13 |
| **Circuit** | 6.0/10 | 7.0/10 | +1.0 | 15% | 1.05 |
| **DPF-Specific** | 3.3/10 | 4.8/10 | +1.5 | 25% | 1.20 |
| **Validation** | 2.3/10 | 2.7/10 | +0.4 | 15% | 0.41 |
| **AI/ML** | 3.0/10 | 3.0/10 | 0 | 5% | 0.15 |
| **Software Eng.** | 7.2/10 | 7.5/10 | +0.3 | 5% | 0.38 |

**Weighted Composite**: 5.6/10

### Panel Positions

- **Dr. PP** (weights circuit/DPF heavily): 5.7/10
  - "Snowplow coupling is the most significant physics improvement since Phase A. L_coeff=2*F_coeff verification removes my primary concern from Debate #4. However, radial back-pressure omission and validation peak error are serious flaws."

- **Dr. DPF** (highest, credits snowplow breakthrough): 6.0/10
  - "This is a qualitative leap. We now have Lee Phases 2-4 implemented with velocity-Verlet integration. The 59% current dip proves the model architecture is sound. The gap to experimental is physics tuning, not fundamental design."

- **Dr. EE** (lowest, penalizes validation heavily): 5.1/10
  - "I cannot award high marks when the primary validation metric uses `np.argmax(np.abs(I_arr))` — finding the global maximum instead of the first peak. This is a junior-level coding error that invalidates all claims about 'within 50% of experimental'."

**Consensus**: 5.6/10 (arithmetic mean of 5.7, 6.0, 5.1)

---

## DETAILED COMPONENT SCORES

### MHD Numerics: 7.75/10 (+0.25)

**Strengths**:
- Metal cylindrical geometry with 7 correct source terms (Stone & Norman 1992):
  - Continuity: `-rho*vr/r`
  - r-momentum: `+(rho*vtheta^2 - Btheta^2/mu_0)/r`
  - theta-momentum: `-(rho*vr*vtheta - Br*Btheta/mu_0)/r`
  - z-momentum: `-(Br*Bz/mu_0)/r`
  - Energy: `-(vr*(E+P) - (v·B)*Br/mu_0)/r`
  - Br-induction: `-(Btheta*vtheta - vtheta*Btheta)/r`
  - Btheta-induction: `+(Br*vtheta - vr*Btheta)/r`
- Dimensional analysis verified for all 7 terms
- 29 cylindrical geometry tests (uniform state preservation, Sod shock stability, energy conservation)

**Weaknesses**:
- First-order operator-split deduction (-0.25 points): cylindrical sources applied after HLLD step, causing formal O(dt) error even with SSP-RK3
- No demonstration of geometric source order >1 (could use Strang splitting for O(dt²))

**Verdict**: Minor improvement over Debate #4 due to cylindrical sources, but operator-split remains a limitation.

---

### Transport: 7.5/10 (no change)

**Status**: No new transport physics in Phases T/U. Score unchanged from Debate #4.

**Existing Strengths**:
- Spitzer resistivity with Braginskii alpha(Z) correction
- Braginskii anisotropic viscosity/conduction (Python engine)
- Nernst advection

**Existing Weaknesses**:
- LHDI anomalous resistivity missing (only ion-acoustic threshold)
- No radiation transport (optically thin approximation only)

---

### Circuit: 7.0/10 (+1.0)

**Major Improvements**:
1. **L_coeff = 2*F_coeff verified** — Critical bug from Debate #4 is FIXED
   - Theoretical derivation: Φ = μ₀*z_f/(2π) * ln(b/a), L = Φ/I
   - Force per unit length: F = μ₀*I²/(4π*r)
   - Verification: ∂L/∂r = -μ₀*z_f/(2π*r) = -2*μ₀*I/(4π*r) = -2*F/I ✓
   - 18 tests added in `test_phase_t_snowplow_coupling.py`

2. **BDF2 dL/dt term** — Backward differentiation formula (2nd-order) for time-varying inductance
   - Code: `dL_dt = (3*L_current - 4*L_history[-1] + L_history[-2]) / (2*dt)`
   - Stability verified for rapidly changing L(t) during pinch

3. **Crowbar implemented** — SCR crowbar with configurable threshold voltage and resistance
   - Activation logic, post-crowbar state, tests present
   - **BUT**: Not enabled in PF-1000 validation runs (see Validation section)

**Remaining Weaknesses**:
- **No parasitic inductance budget** — Assumes all inductance is plasma (ignores feed structure, transmission line)
- **No external load/plasma impedance split** — Treats entire device as single inductance

**Justification**: +1.0 point increase justified by L_coeff=2*F_coeff fix (removes Debate #4 blocker) and BDF2 upgrade. Crowbar implementation adds capability but doesn't count fully since not used in validation.

---

### DPF-Specific Physics: 4.8/10 (+1.5)

**Major Breakthrough**: Current dip signature produced for the first time.

**What's Implemented (Lee Phases 2-4)**:
- **Phase 2 (Axial)**: Snowplow with `F_mag = μ₀*I²/(4π*r)`, pressure drag `F_pressure = p_fill * π*(b²-a²)`, velocity-Verlet integrator
- **Phase 3 (Radial)**: Geometry transition at z=z_max, radial implosion with magnetic force scaling `1/r`, f_c and f_m mass/current fraction parameters
- **Phase 4 (Pinch)**: Minimum radius check, stagnation detection, pinch duration tracking

**Quantitative Results**:
- PF-1000 simulation with default parameters:
  - Current dip: **59%** (1.56 MA → 0.64 MA)
  - Experimental range: **20-40%** (Scholz et al. 2006, Sadowski et al. 2008)
- **Gap**: 19-39 percentage points too deep

**Root Cause Analysis (Panel Consensus)**:
1. **Missing radial adiabatic back-pressure** (Bug N2) — radial implosion too fast
2. **No insulator phase delay** — sheath starts moving 0.5-2 μs too early
3. **Uncalibrated default parameters** — f_m=0.3 vs published 0.08-0.14 for PF-1000
4. **No reflected shock** (Lee Phase 4a) — missing temperature rise during stagnation

**What's NOT Implemented**:
- Lee Phase 1 (insulator): 0.5-2 μs delay before axial acceleration
- Lee Phase 4a (reflected shock): radial bounce, secondary compression, temperature spike
- Lee Phase 5 (final pinch): neutron production, radiative collapse
- Parameter fitting framework: no scipy.optimize calibration vs experimental I(t)

**Scoring Rationale**:
- Base score 3.0 (same as Debate #4)
- +1.5 for current dip production (major qualitative leap)
- +0.3 for velocity-Verlet integrator (2nd-order accurate)
- -0.0 for 59% vs 20-40% gap (not penalized here, covered in Validation)
- **Total**: 4.8/10

**Dr. DPF's Position** (highest): "This is a 5.5/10. The current dip is the Rosetta Stone of DPF validation. Once we add radial back-pressure and insulator phase, we'll match experimental within 20%. The architecture is sound."

**Dr. PP's Position** (moderate): "I agree with 4.8. The dip is impressive, but 59% vs 20-40% is not a rounding error — it's a factor of 1.5-3x. Missing physics is not just tuning."

**Dr. EE's Position** (lowest): "I'd argue 4.5, but I'll accept 4.8. The fact that we can produce *any* dip is notable, but the validation gap is too large to credit more than this."

**Consensus**: 4.8/10 (compromise between 4.5, 4.8, 5.5)

---

### Validation: 2.7/10 (+0.4)

**First-Ever Achievement**: Direct I(t) comparison against PF-1000 experimental data (Scholz et al. 2006).

**Critical Flaw Discovered**:
```python
# experimental.py line 214 — WRONG
peak_idx = np.argmax(np.abs(I_arr))
I_peak_sim = I_arr[peak_idx]
```
This finds the **global maximum** (2.233 MA at 19.18 μs, a post-pinch oscillation artifact), NOT the **first peak before dip** (1.236 MA at 2.16 μs).

**Actual vs Claimed Error**:
- **Claimed** (using global max): "within 50%" tolerance → PASS
- **Actual** (using first peak): 33.9% error (1.236 MA vs 1.85 MA experimental) → FAIL

**Impact on All Validation Claims**:
- `validate_current_waveform()` uses this metric for PF-1000, NX2, and UNU-ICTP
- All current validation results are **invalid**
- This is a **junior-level coding error**, not a physics disagreement

**Additional Validation Gaps**:
1. **No uncertainty budget** — experimental data has ±10-15% shot-to-shot variation (Scholz), not accounted for
2. **50% tolerance is order-of-magnitude** — not quantitative validation (should be <15% for PhD-level)
3. **No waveform shape comparison** — only checks single peak value, ignores dip depth, dip timing, rise time, etc.
4. **Crowbar not enabled** — post-pinch oscillations are non-physical artifacts from inductive ringing

**What's Present (Partial Credit)**:
- 55 ablation V&V tests (mass flux, energy balance, pressure equilibrium)
- Sod/Brio-Wu verification tests (but no quantitative convergence)
- Cross-backend parity tests

**Scoring Rationale**:
- +0.4 for first-ever I(t) comparison attempt (bold effort, even if flawed)
- +1.0 for ablation V&V coverage
- +0.3 for cross-backend tests
- +1.0 for shock tube stability
- **-0.5 for wrong peak metric** (invalidates primary result)
- **-0.5 for no uncertainty budget**
- Base from Debate #4: 2.3/10
- **Total**: 2.7/10

**Dr. EE's Position** (lowest): "This should be 2.0/10. Using `np.argmax(np.abs(I_arr))` is inexcusable. I teach undergrads to find local maxima with `scipy.signal.find_peaks()`. This invalidates the entire PF-1000 comparison."

**Dr. DPF's Position** (highest): "I'll accept 2.7. The error is bad, but the *attempt* at experimental comparison is a huge step forward. Most academic codes never get this far."

**Dr. PP's Position** (moderate): "Agreed at 2.7. The crowbar issue is equally serious — those post-pinch oscillations are not physics, they're circuit artifacts."

**Consensus**: 2.7/10

---

### AI/ML: 3.0/10 (no change)

No changes in Phases T/U. Score unchanged from Debate #4.

---

### Software Engineering: 7.5/10 (+0.3)

**Test Coverage**:
- **Phase T**: 133 tests (snowplow coupling, inductance derivatives, BDF2, crowbar)
- **Phase U**: 84 tests (cylindrical sources, ablation V&V, Metal geometry)
- **Total**: 1965 tests (1837 non-slow, 128 slow)
- **All passing** (verified 2026-02-17)

**Code Quality**:
- Clean separation: `snowplow.py` (Lee model), `rlc_solver.py` (circuit), `cylindrical_mhd.py` (geometry)
- Type hints, NumPy-style docstrings, Ruff compliance
- Pydantic configuration

**Minor Weaknesses**:
- `lee_model_comparison.py` has naming swap (fm/fc) and missing z_f factor (Bugs N3, N4)
- Some validation tests use loose tolerances (50%)

**Justification**: +0.3 for 217 new tests and clean architecture.

---

## CONFIRMED BUGS

### Total: 6 (3 inherited, 3 new)

### From Previous Debates (Still Open)

**Bug 1: Non-conservative pressure in Python engine** (inherited from Debate #1)
**Severity**: MODERATE (mitigated by demotion to teaching-only in Phase R)
**Location**: `src/dpf/fluid/mhd_solver.py`
**Impact**: Python engine uses `np.gradient` for pressure gradient, violating conservation in shocks
**Status**: MITIGATED (not fixed) — Python engine demoted to teaching tier

**Bug 2: LHDI anomalous resistivity missing**
**Severity**: HIGH
**Location**: `src/dpf/collision/anomalous.py`
**Impact**: Only ion-acoustic threshold implemented, missing Lower-Hybrid Drift Instability (dominant in DPF sheared flows)
**Status**: OPEN — awaiting Phase V+ implementation

---

### New Bugs (Debate #5)

**Bug 3: validate_current_waveform uses wrong peak** ⚠️ **CRITICAL**
**Severity**: CRITICAL
**Location**: `src/dpf/validation/experimental.py` line 214
**Description**:
```python
# WRONG — finds global maximum (post-pinch oscillation)
peak_idx = np.argmax(np.abs(I_arr))
I_peak_sim = I_arr[peak_idx]
```
Should find **first local maximum before dip**:
```python
# CORRECT
from scipy.signal import find_peaks
peaks, _ = find_peaks(np.abs(I_arr))
peak_idx = peaks[0]  # first peak
I_peak_sim = I_arr[peak_idx]
```

**Impact**:
- All PF-1000/NX2/UNU-ICTP validation claims invalid
- Example: PF-1000 claims "within 50%" using 2.233 MA (global max at 19.18 μs)
- Actual first peak: 1.236 MA (2.16 μs) → 33.9% error vs 1.85 MA experimental

**Confirmed by**: All 3 panelists (unanimous)

**Fix Effort**: 1 day (add scipy.signal.find_peaks, update test baselines)

---

**Bug 4: Missing radial adiabatic back-pressure in snowplow** ⚠️ **HIGH**
**Severity**: HIGH
**Location**: `src/dpf/atomic/snowplow.py` line 374 (in `_step_radial`)
**Description**:
```python
# Current code — F_pressure = 0.0 for radial phase
if self.phase == SnowplowPhase.RADIAL:
    return F_mag  # only magnetic force
```
Lee model requires adiabatic compression back-pressure:
```python
p_adiabatic = p_fill * (b / r_s)**(2 * self.config.gamma)
F_pressure = p_adiabatic * 2 * np.pi * r_s * z_f
return F_mag - F_pressure  # subtract back-pressure
```

**Physics**: As sheath implodes radially, trapped gas compresses adiabatically. Without this, radial implosion too fast → 59% current dip vs 20-40% experimental.

**Note**: Axial phase F_pressure IS implemented correctly (`p_fill * π*(b²-a²)`).

**Impact**:
- Current dip 59% vs 20-40% experimental (19-39 pp too deep)
- Radial velocity ~3x too fast during implosion
- Pinch radius ~2x too small at stagnation

**Confirmed by**: All 3 panelists

**Fix Effort**: 2 days (add p_adiabatic term, recalibrate f_m/f_c, update validation baselines)

---

**Bug 5: fm/fc naming swap in lee_model_comparison.py** ⚠️ **MODERATE**
**Severity**: MODERATE (low numerical impact, high confusion risk)
**Location**: `src/dpf/validation/lee_model_comparison.py` lines 196-198
**Description**:
```python
# WRONG naming
self.fm = current_fraction  # should be mass_fraction
self.fc = mass_fraction     # should be current_fraction
```
Lee's notation: f_m = mass fraction, f_c = current fraction.
Production code (`snowplow.py`) uses CORRECT naming.

**Impact**:
- No numerical error at default values (both 0.7)
- High confusion risk when comparing to literature
- Affects cross-check module only, not production physics

**Confirmed by**: All 3 panelists

**Fix Effort**: 1 hour (swap lines, update docstrings)

---

**Bug 6: lee_model_comparison.py radial force missing z_f factor** ⚠️ **MODERATE**
**Severity**: MODERATE
**Location**: `src/dpf/validation/lee_model_comparison.py` line 394
**Description**:
```python
# Current code — force per unit length
F_rad = (mu_0 / (4 * np.pi)) * (self.fm * I)**2 / r

# Correct — total force
F_rad = (mu_0 / (4 * np.pi)) * (self.fm * I)**2 * z_f / r
```

**Physics**: μ₀*I²/(4π*r) is force **per unit length**. Total force requires multiplying by sheath height z_f.

**Impact**:
- Radial acceleration wrong by factor of 1/z_f
- For PF-1000 (z_f ~ 0.16 m at transition), error is 6.25x
- Causes incorrect Lee model comparison in cross-check tests
- **NOTE**: Production `snowplow.py` line 339 is CORRECT (includes z_f)

**Confirmed by**: Dr. PP (Phase 2 cross-examination)

**Fix Effort**: 1 hour (add `* self.z_f` to line 394)

---

### Retracted Claims

**Dr. PP's Bug N4 (initial_energy)**: Confirmed as naming inconsistency but dead code in main loop — reduced to LOW severity, not counted in bug list.

**Dr. PP's Bug N5 (operator-split)**: Reclassified as standard design choice. Operator splitting is O(dt) accurate but widely used in production codes (FLASH, Athena++). **RETRACTED** as bug, remains as MHD Numerics deduction (-0.25 points).

**Dr. DPF's speed factor S=48.6**: Claimed sheath velocity 48.6x too slow. Dr. EE identified unit conversion error (cm/s vs m/s). **RETRACTED**.

---

## KEY FINDINGS

### What Improved (Phases T + U)

1. **Snowplow model exists and works** ✓
   - Lee Phases 2-4 implemented (axial, radial, pinch)
   - Velocity-Verlet integrator (2nd-order accurate)
   - f_c and f_m parameters for current/mass fraction
   - Geometry transition logic (axial → radial at z=z_max)

2. **Current dip signature produced** ✓
   - First time ever for this simulator
   - 59% depth (vs 20-40% experimental)
   - Demonstrates model architecture is sound

3. **L_coeff = 2*F_coeff verified correct** ✓
   - Critical bug from Debate #4 is FIXED
   - Theoretical derivation matches code
   - 18 verification tests added

4. **Metal cylindrical geometry** ✓
   - 7 correct source terms (Stone & Norman 1992)
   - 29 tests (uniform state, Sod stability, energy conservation)
   - Dimensional analysis verified

5. **Ablation V&V** ✓
   - 55 tests covering all ablation functions
   - Mass flux, energy balance, pressure equilibrium

6. **Test coverage growth** ✓
   - 1965 total tests (up from ~1600 after Phase S)
   - 217 new tests in Phases T+U
   - All passing

---

### What Still Needs Work

#### Validation (2.7/10 → Target 6.0/10)

1. **Fix validate_current_waveform to find first peak** (Bug 3)
   - Use `scipy.signal.find_peaks()` instead of `np.argmax(np.abs())`
   - Add dip_depth, dip_time, first_peak_time metrics
   - **Effort**: 1 day

2. **Add uncertainty budget**
   - PF-1000 has ±10-15% shot-to-shot variation (Scholz 2006)
   - Simulation should report confidence intervals
   - **Effort**: 2 days

3. **Tighten tolerances**
   - 50% is order-of-magnitude, not quantitative
   - Target: <15% I_peak error, <20% timing error
   - **Effort**: Requires fixing Bugs 4-6 first

4. **Enable crowbar for validation runs**
   - Post-pinch oscillations are circuit artifacts, not physics
   - **Effort**: 1 day (config change + baseline update)

#### DPF-Specific Physics (4.8/10 → Target 7.0/10)

5. **Add radial adiabatic back-pressure** (Bug 4)
   - p_adiabatic = p_fill * (b/r_s)^(2*gamma)
   - Should reduce dip from 59% → 30-40%
   - **Effort**: 2 days

6. **Add insulator phase (Lee Phase 1)**
   - 0.5-2 μs delay before axial acceleration
   - Shifts all timing, improves I_peak match
   - **Effort**: 1 day

7. **Use published PF-1000 parameters**
   - f_m = 0.08-0.14 (not default 0.3)
   - f_c = 0.7 (correct)
   - Scholz et al. (2006) Table II
   - **Effort**: 1 day (config change)

8. **Implement parameter fitting framework**
   - Use `scipy.optimize.minimize` to fit f_m, f_c, f_mr, t_delay to experimental I(t)
   - Minimize residual: Σ|I_sim(t) - I_exp(t)|²
   - **Effort**: 1 week

9. **Add reflected shock model (Lee Phase 4a)**
   - Radial bounce at stagnation
   - Secondary compression, temperature spike
   - Required for neutron yield prediction
   - **Effort**: 1 week

#### Minor Fixes

10. **Fix fm/fc naming in lee_model_comparison.py** (Bug 5)
    - **Effort**: 1 hour

11. **Fix missing z_f in lee_model_comparison.py radial force** (Bug 6)
    - **Effort**: 1 hour

---

## PATH TO 7.0/10 COMPOSITE

The panel agrees the following would raise the score to **~7.0/10**:

### Phase V Roadmap (2-3 weeks total)

#### Week 1: Validation Infrastructure
- Fix `validate_current_waveform` to find first peak (Bug 3) — 1 day
- Add uncertainty budget to validation comparisons — 2 days
- Enable crowbar for PF-1000 validation runs — 1 day
- Fix fm/fc naming and missing z_f (Bugs 5, 6) — 2 hours
- **Expected**: Validation → 5.0/10

#### Week 2: DPF Physics Improvements
- Add radial adiabatic back-pressure (Bug 4) — 2 days
- Add insulator phase delay parameter — 1 day
- Use published PF-1000 parameters (f_m=0.08-0.14) — 1 day
- Recalibrate and re-run validation suite — 1 day
- **Expected**: DPF-Specific → 6.5/10, Validation → 6.0/10

#### Week 3: Parameter Calibration
- Implement `scipy.optimize` fitting of f_m, f_c, f_mr, t_delay to match I(t) — 3 days
- Validate against PF-1000 with fitted parameters (<15% I_peak error) — 2 days
- Cross-validate against NX2 and UNU-ICTP — 2 days
- **Expected**: DPF-Specific → 7.5/10, Validation → 7.0/10

### Projected Scores After Phase V

| Component | Current | After Week 1 | After Week 2 | After Week 3 |
|-----------|---------|--------------|--------------|--------------|
| MHD Numerics | 7.75 | 7.75 | 7.75 | 7.75 |
| Transport | 7.5 | 7.5 | 7.5 | 7.5 |
| Circuit | 7.0 | 7.0 | 7.0 | 7.0 |
| DPF-Specific | 4.8 | 4.8 | 6.5 | 7.5 |
| Validation | 2.7 | 5.0 | 6.0 | 7.0 |
| AI/ML | 3.0 | 3.0 | 3.0 | 3.0 |
| Software Eng. | 7.5 | 7.5 | 7.5 | 7.5 |
| **Composite** | **5.6** | **6.2** | **6.8** | **7.1** |

---

## CONSENSUS VERIFICATION CHECKLIST

- [x] **Mathematical derivations provided**
  - Snowplow force: F = μ₀*I²/(4π*r) per unit length
  - Inductance: L = μ₀*z_f/(2π) * ln(b/a)
  - Inductance derivative: ∂L/∂r = -2*F/I
  - Cylindrical sources: all 7 terms verified (Stone & Norman 1992)
  - Dimensional analysis: SI-consistent (except lee_model_comparison.py Bug 6)

- [x] **3+ peer-reviewed citations**
  - Lee & Saw (2014): Snowplow model Phases 1-5
  - Scholz et al. (2006): PF-1000 parameters, experimental I(t)
  - Stone & Norman (1992): Cylindrical source terms
  - Mignone et al. (2007): HLLD Riemann solver
  - Sahyouni et al. (2021): Lee model validation vs NX2
  - Sadowski et al. (2008): PF-1000 current dip 20-40%

- [ ] **Experimental evidence cited**
  - I(t) comparison attempted vs PF-1000 (Scholz 2006)
  - **BUT**: Methodology flawed (wrong peak metric, Bug 3)
  - Partial credit for attempt

- [x] **All assumptions explicitly listed**
  - Dr. DPF provided 12 assumptions with regime of validity:
    1. Single-fluid MHD (valid for ωce*τei > 1)
    2. Snowplow approximation (valid for dense sheath, L >> thickness)
    3. No reflected shock (invalid for stagnation, Lee Phase 4a needed)
    4. Constant gamma (invalid for ionization/radiation, EOS coupling needed)
    5. Uniform pressure behind sheath (valid for subsonic flow)
    6. No skin depth effects (valid for L >> c/ωpe)
    7. Current uniformity f_c (empirical fit, not first-principles)
    8. Mass pickup f_m (empirical fit, needs calibration)
    9. Insulator phase neglected (invalid for t < 0.5-2 μs)
    10. Circuit as lumped element (valid for τ >> L/c)
    11. No crowbar in validation baseline (invalid for t > t_pinch)
    12. Adiabatic radial compression (implemented in axial, missing in radial — Bug 4)

- [ ] **Uncertainty budget**
  - **NOT present** in validation metrics
  - Experimental data has ±10-15% shot-to-shot variation (Scholz)
  - Simulation reports point estimates only

- [x] **All cross-examination criticisms addressed**
  - 15 challenges resolved across Phases 2-3:
    - Dr. PP: L_coeff vs F_coeff verified (Debate #4 blocker removed)
    - Dr. EE: sigma analysis qualified (sheath acceleration, not axial)
    - Dr. DPF: speed factor error retracted (unit conversion)
    - Dr. PP: Bug N4 (initial_energy) confirmed as LOW severity
    - Dr. PP: Bug N5 (operator-split) reclassified, not a bug
    - Dr. PP: Bug N6 (missing z_f) confirmed as MODERATE (lee_model_comparison.py only)
    - All 3: Bugs 3-4 confirmed CRITICAL/HIGH

- [x] **No unresolved logical fallacies**
  - Dr. DPF's speed factor argument retracted after unit error
  - Dr. EE's sigma analysis qualified (correct for sheath, not bulk plasma)
  - Dr. PP's operator-split concern reclassified as design choice

- [x] **Explicit agreement from each panelist**
  - Dr. PP: 5.7/10
  - Dr. DPF: 6.0/10
  - Dr. EE: 5.1/10
  - Consensus: 5.6 ± 0.5

### Dissenting Notes (within consensus)

**Dr. EE** (lowest scorer):
> "I argue Validation should be 2.0/10, not 2.7. Using `np.argmax(np.abs())` is a junior-level error that invalidates all claims. I'm being generous at 2.7. This would lower composite to ~5.3."

**Dr. DPF** (highest scorer):
> "I argue DPF-Specific should be 5.5/10, not 4.8. Producing the current dip is a qualitative leap. The 59% vs 20-40% gap is fixable with Bug 4 (radial back-pressure) and insulator phase. This would raise composite to ~5.9."

**Consensus Moderator Note**:
Both positions documented. Consensus uses compromise values (2.7, 4.8) as middle ground. Final composite 5.6 represents the arithmetic mean of panel scores (5.7, 6.0, 5.1), not a weighted average of dissenting positions.

---

## RECOMMENDATIONS FOR PHASE V

### Priority 1: Fix Validation Infrastructure (1-2 days)

**Goal**: Raise Validation score from 2.7 → 5.0

1. **Fix validate_current_waveform** (Bug 3)
   - Replace `np.argmax(np.abs(I_arr))` with `scipy.signal.find_peaks()`
   - Find first local maximum before dip
   - Add test: first_peak < dip_time < global_max

2. **Add waveform shape metrics**
   - `dip_depth = 1 - I_min/I_peak` (target: 20-40%)
   - `dip_time` (target: 3-5 μs for PF-1000)
   - `first_peak_time` (target: 2-3 μs)
   - `rise_time` (10-90% of I_peak)

3. **Add uncertainty budget**
   - Report shot-to-shot variation from literature (±10-15% for PF-1000)
   - Report simulation sensitivity to f_m, f_c variations
   - Use error bars in all plots

4. **Enable crowbar**
   - Set `enable_crowbar=True` for all validation runs
   - Crowbar threshold: 1.5*V_peak (typical SCR rating)
   - Crowbar resistance: 0.01 Ω (SCR + transmission line)

---

### Priority 2: Fix Radial Phase Physics (2-3 days)

**Goal**: Raise DPF-Specific score from 4.8 → 6.5

1. **Add adiabatic back-pressure** (Bug 4)
   ```python
   # In snowplow.py, _step_radial():
   if self.phase == SnowplowPhase.RADIAL:
       p_adiabatic = self.p_fill * (self.b / self.r_s)**(2 * self.config.gamma)
       A_annular = 2 * np.pi * self.r_s * self.z_f
       F_pressure = p_adiabatic * A_annular
       return F_mag - F_pressure  # subtract back-pressure
   ```

2. **Add insulator phase** (Lee Phase 1)
   - Add `t_insulator_delay` parameter (default: 0.5 μs)
   - During delay: F_mag = F_pressure = 0, sheath stationary
   - Transition to axial phase when t > t_delay

3. **Use published PF-1000 parameters**
   - `f_m = 0.12` (midpoint of 0.08-0.14 from Scholz Table II)
   - `f_c = 0.7` (unchanged)
   - `gamma = 1.4` (unchanged)

4. **Minor fixes**
   - Fix fm/fc naming in lee_model_comparison.py (Bug 5)
   - Add missing z_f factor in lee_model_comparison.py (Bug 6)

---

### Priority 3: Parameter Calibration (1 week)

**Goal**: Raise DPF-Specific to 7.5, Validation to 7.0

1. **Implement fitting framework**
   ```python
   from scipy.optimize import minimize

   def residual(params, I_exp, t_exp):
       f_m, f_c, f_mr, t_delay = params
       config.snowplow.f_m = f_m
       config.snowplow.f_c = f_c
       config.snowplow.t_insulator = t_delay
       summary = engine.run(...)
       I_sim = summary["current_history"]
       t_sim = summary["time_history"]
       I_sim_interp = np.interp(t_exp, t_sim, I_sim)
       return np.sum((I_sim_interp - I_exp)**2)

   result = minimize(residual, x0=[0.12, 0.7, 0.1, 0.5e-6],
                     bounds=[(0.05, 0.2), (0.5, 0.9), (0.05, 0.2), (0, 2e-6)])
   ```

2. **PF-1000 calibration**
   - Fit against Scholz et al. (2006) I(t) waveform
   - Target: <15% I_peak error, <20% dip_depth error, <10% timing error
   - Report fitted parameters with uncertainties

3. **Cross-validation**
   - Use fitted PF-1000 parameters on NX2 (Sahyouni 2021)
   - Use fitted PF-1000 parameters on UNU-ICTP (Damideh 2017)
   - If errors >30%, indicates physics missing (not just parameters)

---

### Priority 4: Post-Pinch Physics (1 week) — FUTURE

**Goal**: Enable neutron yield prediction (not required for 7.0/10)

1. **Implement reflected shock** (Lee Phase 4a)
   - Detect stagnation: dr/dt → 0
   - Apply radial bounce: v_r → -v_r * restitution_coeff
   - Secondary compression: p_pinch → p_pinch * compression_factor
   - Temperature spike: T_i ~ (m_i/k_B) * v_r²

2. **Add neutron yield estimator**
   - Use Bosch-Hale D-D fusion cross-section
   - n_yield ∝ n_D² * σv * V_pinch * τ_pinch
   - Validate against PF-1000 experimental yield (10⁸-10⁹ per shot)

3. **Voltage waveform validation**
   - Compare V(t) against experimental (Scholz Fig. 3)
   - Check voltage reversal timing (should coincide with I_min)

---

## CONCLUSION

Phases T and U represent the **first qualitative leap in DPF-specific physics** since project inception. The snowplow model produces a current dip signature, proving the model architecture is sound. However, the 59% dip depth vs 20-40% experimental reveals missing physics (radial back-pressure, insulator phase), and the validation methodology is fundamentally flawed (wrong peak metric, loose tolerances, no uncertainty budget).

The simulator remains **bifurcated**: strong generic MHD infrastructure (7.5+) but weak DPF validation (2.7). The path to 7.0/10 is clear and achievable in 2-3 weeks:

1. **Week 1**: Fix validation infrastructure → 6.2/10 composite
2. **Week 2**: Add radial back-pressure and insulator phase → 6.8/10 composite
3. **Week 3**: Implement parameter fitting framework → 7.1/10 composite

This is not a research problem — it's an engineering checklist. The physics is known (Lee 2014, Scholz 2006), the code architecture is ready, and the test coverage is strong. **The next debate should target 7.0+ composite, achieved through systematic bug fixing and validation tightening, not new physics capabilities.**

---

**Signatures** (Consensus 3-0):

- Dr. PP (Pulsed Power): 5.7/10
  *"Snowplow coupling is impressive. Fix Bugs 3-4 and we're at 7.0."*

- Dr. DPF (Plasma Physics): 6.0/10
  *"Current dip is the Rosetta Stone. Architecture is sound. Now tune the parameters."*

- Dr. EE (Electrical Engineering): 5.1/10
  *"Cannot award high marks when `np.argmax(np.abs())` is the validation metric. Fix this first."*

**Consensus**: 5.6/10 (arithmetic mean: (5.7 + 6.0 + 5.1)/3 = 5.6)

---

**Next Review**: After Phase V (Validation Infrastructure + Radial Physics)
**Target Composite**: 7.0/10
**Expected Timeline**: 2-3 weeks (Feb-Mar 2026)
