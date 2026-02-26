# Validation Package Troubleshooting

**Cross-review by**: xreview-circuit (synthesizing py-circuit-val + phys-circuit-val)
**Files reviewed**: `bennett_equilibrium.py` (243 LOC), `calibration.py` (418 LOC), `experimental.py` (523 LOC), `lee_model_comparison.py` (680 LOC), `magnetized_noh.py` (404 LOC), `suite.py` (615 LOC), `__init__.py` (87 LOC)
**Date**: 2026-02-25

---

## CRITICAL Findings

### CRIT-1: Two `normalized_rmse` functions with different normalization — namespace collision ✅ FIXED

- **File**: `validation/__init__.py:47`, `experimental.py:253-284`, `suite.py:195-218`
- **Found by**: py-circuit-val (CC-1, V-EXP-2)
- **Cross-review verdict**: **CONFIRMED** — verified in `__init__.py` source
- **Fix**: Renamed to `nrmse_peak` (experimental) and `nrmse_range` (suite). Both exported from `__init__.py` with distinct names. Backward-compatible `normalized_rmse` aliases preserved in each module.

**Description**: Two different `normalized_rmse` functions exist:
1. `experimental.py:253-284`: `NRMSE = RMSE / max(|I_peak_exp|)` (peak-normalized, 4-argument waveform comparison)
2. `suite.py:195-218`: `NRMSE = RMSE / (max(ref) - min(ref))` (range-normalized, 2-argument array comparison)

The `__init__.py` imports `normalized_rmse` from `suite.py` (line 47), which shadows the `experimental.py` version. A caller doing `from dpf.validation import normalized_rmse` gets the range-normalized version. However, `lee_model_comparison.py:548` correctly imports `normalized_rmse` from `experimental` directly, so the waveform comparison chain is unaffected.

**Evidence**:
```python
# __init__.py:47 — last import wins
from dpf.validation.suite import (
    ...
    normalized_rmse,  # This shadows experimental.normalized_rmse
    ...
)
```

**Correct behavior**: Two functions with the same name but different semantics should have distinct names.

**Proposed fix**: Rename `suite.normalized_rmse` to `nrmse_range` and `experimental.normalized_rmse` to `nrmse_peak`, then update `__init__.py` to export both with their distinct names. Alternatively, only export one and require direct imports for the other.

**Impact**: Currently no runtime bug because `lee_model_comparison.py` imports directly from `experimental`. But any new code using the package-level import will get the wrong function for waveform comparison.

### CRIT-2: Root cause of anomalous fc=0.5, fm=0.95 — 2-phase Lee model is fundamentally insufficient ✅ FIXED (bounds tightened)

- **File**: `calibration.py:89-90`, `lee_model_comparison.py` (entire file)
- **Found by**: phys-circuit-val (BUG-CV-2)
- **Cross-review verdict**: **CONFIRMED** — this is the definitive root cause analysis
- **Fix**: Tightened defaults to `fc_bounds=(0.65, 0.85)`, `fm_bounds=(0.05, 0.25)` per Lee & Saw (2014). Also added native `scipy.optimize.Bounds` (MED-1) and renormalized weights when waveform unavailable (MED-3).

**Description**: The calibrator produces `fc=0.5, fm=0.95` for PF-1000, far outside the published Lee & Saw (2014) ranges of `fc in [0.65, 0.75], fm in [0.05, 0.15]`. The physics reviewer identified three interacting causes:

1. **Missing Lee model phases 3-5**: `lee_model_comparison.py` implements only phases 1-2 (axial + radial). It does NOT implement:
   - Phase 3b: Reflected shock dynamics
   - Phase 4: Slow compression (thermodynamic pinch)
   - Phase 5: Instability disruption and beam formation

   The published fc/fm values from Lee & Saw were obtained with the full 5-phase model. The 2-phase model has insufficient physics to constrain these parameters.

2. **Overly wide bounds**: `fc_bounds=(0.5, 0.95)` and `fm_bounds=(0.05, 0.95)` allow the optimizer to explore physically unreasonable regions. The upper fm bound of 0.95 means 95% of the fill gas is swept — only realistic in a 5-phase model where reflections redistribute mass.

3. **Waveform NRMSE drives optimizer to boundary**: The objective includes a 30% weight on waveform NRMSE, which is dominated by the post-pinch current trace. The 2-phase model cannot reproduce post-pinch dynamics, so the optimizer pushes fm to extreme values to fit the I(t) decay.

**Evidence**:
```python
# calibration.py:89-90
fc_bounds: tuple[float, float] = (0.5, 0.95),
fm_bounds: tuple[float, float] = (0.05, 0.95),

# lee_model_comparison.py:8-9
# This implementation covers phases 1 and 2 only (MVP)
```

**Correct behavior**: Either (a) implement the full 5-phase Lee model (~500 LOC additional), (b) tighten bounds to match published ranges (`fm_bounds=(0.05, 0.25)`), or (c) use only the pre-dip portion of I(t) for calibration objective.

**Impact**: The calibrated fc/fm values are physically meaningless. The `benchmark_against_published()` method correctly reports this discrepancy, but downstream code that trusts the calibration is affected.

---

## HIGH Findings

### HIGH-1: Radial phase acceleration clamped to non-positive prevents reflected shock ✅ FIXED

- **File**: `lee_model_comparison.py:417`
- **Found by**: py-circuit-val (V-LEE-2)
- **Cross-review verdict**: **CONFIRMED** — this is a deliberate 2-phase simplification but limits model validity
- **Fix**: Removed `vr = min(vr, 0.0)` clamp and changed `np.clip(dvr_dt, -1e15, 0.0)` to symmetric `np.clip(dvr_dt, -1e15, 1e15)`. Shock can now decelerate and bounce via adiabatic back-pressure.

**Description**: The radial phase ODE clips `dvr_dt` to be non-positive:
```python
dvr_dt = np.clip(dvr_dt, -1e15, 0.0)
```
Combined with the velocity clamp at line 381 (`vr = min(vr, 0.0)`), the shock can never decelerate, stagnate, or bounce outward. This prevents any physical bounce/reflection, which is the mechanism that creates the current dip in real DPF waveforms.

**Evidence**: Lines 381 and 417 prevent any outward (positive vr) motion.

**Impact**: The model produces I(t) waveforms without a proper current dip, which is the primary observable signature of pinch. This feeds into CRIT-2: the optimizer compensates for the missing dip by pushing fm to extreme values.

### HIGH-2: Neutron yield uses peak current instead of pinch current ✅ FIXED

- **File**: `lee_model_comparison.py:638`
- **Found by**: py-circuit-val (V-LEE-4), acknowledged by phys-circuit-val as "acceptable for order-of-magnitude"
- **Cross-review verdict**: **CONFIRMED** — systematic overestimate
- **Fix**: Now uses `result.I[pinch_idx]` via `np.searchsorted(result.t, result.pinch_time)`. Falls back to peak current if pinch_time is unavailable.

**Description**: `I_pinch = max(result.peak_current, 0.0)` uses the peak current (which occurs during the axial phase, before the current dip) rather than the actual current at the moment of pinch (typically 60-80% of peak). Since neutron yield scales as ~I_pinch^4 (through B^2 * v_A * tau dependencies), using peak current overestimates yield by a factor of ~2-6x.

**Evidence**:
```python
# lee_model_comparison.py:638
I_pinch = max(result.peak_current, 0.0)
```

**Correct behavior**: Use `result.I[pinch_time_idx]` where `pinch_time_idx` is the index at `t = pinch_time`. The pinch time is available in `result.pinch_time`.

**Proposed fix**:
```python
# Find current at pinch time
if result.pinch_time > 0 and len(result.t) > 0:
    pinch_idx = np.searchsorted(result.t, result.pinch_time)
    pinch_idx = min(pinch_idx, len(result.I) - 1)
    I_pinch = max(abs(result.I[pinch_idx]), 0.0)
else:
    I_pinch = max(result.peak_current, 0.0)  # fallback
```

**Impact**: Neutron yield overestimated by factor ~2-6x. For an order-of-magnitude estimate (stated accuracy of the 0D model), this is within the expected uncertainty but still a systematic bias.

---

## MEDIUM Findings

### MED-1: Nelder-Mead with manual clamping instead of native bounds ✅ FIXED

- **File**: `calibration.py:117-119, 164-165`
- **Found by**: py-circuit-val (V-CAL-1)
- **Cross-review verdict**: **CONFIRMED** — Nelder-Mead supports `bounds=` since SciPy 1.7.0 (2021)
- **Fix**: Now uses `scipy.optimize.Bounds` in the `minimize()` call. Manual clamping removed from `_objective()`.

**Description**: The code manually clamps parameters inside the objective function, creating flat plateaus at boundaries that confuse the simplex geometry. Since SciPy 1.7.0, `scipy.optimize.minimize(method='Nelder-Mead')` natively supports the `bounds=` parameter.

**Evidence**:
```python
# calibration.py:117-119
self._fc_bounds = fc_bounds
self._fm_bounds = fm_bounds
# calibration.py:164-165
fc = float(np.clip(params[0], *self._fc_bounds))
fm = float(np.clip(params[1], *self._fm_bounds))
```

**Proposed fix**:
```python
from scipy.optimize import Bounds
result = minimize(
    self._objective,
    x0_arr,
    method=self.method,
    bounds=Bounds([fc_bounds[0], fm_bounds[0]], [fc_bounds[1], fm_bounds[1]]),
    options={"maxiter": maxiter, "xatol": 0.005, "fatol": 0.001},
)
```

**Impact**: The optimizer wastes evaluations probing flat plateaus. With native bounds it would converge faster and avoid boundary artifacts.

### MED-2: Bare `Exception` catch silently masks bugs ✅ FIXED

- **File**: `calibration.py:169-171`
- **Found by**: py-circuit-val (V-CAL-2)
- **Cross-review verdict**: **CONFIRMED** — should narrow to specific exceptions
- **Fix**: Narrowed to `except (RuntimeError, ValueError, FloatingPointError):`.

**Description**: Catching `Exception` (including `TypeError`, `KeyError`, `AttributeError`) masks bugs in the Lee model and returns a penalty of 10.0. The optimizer can't distinguish a real high-error region from a crash region.

**Evidence**:
```python
# calibration.py:169-171
except Exception:
    logger.debug("Objective evaluation failed for fc=%.3f, fm=%.3f", fc, fm)
    return 10.0
```

**Proposed fix**: `except (RuntimeError, ValueError, FloatingPointError):` — these are the expected failure modes of `solve_ivp` and the Lee model physics.

### MED-3: Objective weights not renormalized when waveform unavailable ✅ FIXED

- **File**: `calibration.py:67-68, 173-182`
- **Found by**: py-circuit-val (V-CAL-3)
- **Cross-review verdict**: **CONFIRMED** — weights sum to 0.7 when waveform is missing
- **Fix**: Weights are now renormalized to sum to 1.0 when waveform NRMSE is NaN.

**Description**: Default weights are `(0.4, 0.3, 0.3)`. If `waveform_nrmse` is NaN (no waveform data), the waveform term is dropped but weights aren't renormalized. The effective objective becomes `0.4*peak + 0.3*timing`, summing to 0.7. Devices with waveform data get a maximum objective of 1.0, while devices without get max 0.7. This makes objective values non-comparable across devices.

**Proposed fix**: Renormalize weights when waveform is unavailable:
```python
if not (isinstance(nrmse, (int, float)) and np.isfinite(nrmse)):
    total = self.peak_weight + self.timing_weight
    obj = (self.peak_weight / total) * comparison.peak_current_error \
        + (self.timing_weight / total) * comparison.timing_error
```

### MED-4: `_find_first_peak` returns last point of a flat-topped peak

- **File**: `experimental.py:237-243`
- **Found by**: py-circuit-val (V-EXP-1)
- **Cross-review verdict**: **PARTIALLY CONFIRMED** — edge case, unlikely to occur with digitized DPF waveforms

**Description**: For a flat-topped signal `[..., 1.87, 1.87, 1.87, 1.85, ...]`, the condition `signal[i] >= signal[i-1] and signal[i] >= signal[i+1]` is satisfied at the **last** plateau point (where `signal[i] >= signal[i+1]` first becomes strict inequality), not the first. However, DPF current waveforms from Rogowski coils have sufficient noise that flat-topped peaks are extremely rare in practice.

**Impact**: Low. The index difference between first and last plateau point is typically 0-1 points, which for 26-point digitized waveforms represents ~0.5 us — within the 10% timing uncertainty.

### MED-5: PF-1000 timing data inconsistency across modules

- **File**: `suite.py:89` vs `experimental.py:138`
- **Found by**: py-circuit-val (V-SUI-1), phys-circuit-val (minor inconsistency)
- **Cross-review verdict**: **CONFIRMED** — 5.5 us vs 5.8 us

**Description**: `suite.py` uses `peak_current_time_s=5.5e-6` while `experimental.py` uses `current_rise_time=5.8e-6`. These represent different quantities (actual peak time vs quarter-period), but the 5% discrepancy creates ambiguity. Additionally, `suite.py` uses `L0=33.5e-9` vs `experimental.py` uses `L0=33e-9` (1.5% difference).

**Impact**: The calibration module (`lee_model_comparison.py`) uses `experimental.DEVICES` first (line 135), so the 5.8 us value is used consistently for calibration. The `suite.py` values only affect `ValidationSuite.validate_circuit()`. Both are within measurement uncertainty but should ideally be consistent.

### MED-6: Axial velocity lost at radial phase transition

- **File**: `lee_model_comparison.py:363`
- **Found by**: py-circuit-val (V-LEE-3)
- **Cross-review verdict**: **CONFIRMED** — acknowledged as standard Lee model simplification

**Description**: `vr_start = 0.0` is hardcoded. The axial velocity `_vz1` at the end of phase 1 (line 334) is computed but never used. The kinetic energy of the axial motion is lost in the phase transition.

**Evidence**:
```python
# line 334
_vz1 = sol1.y[3]  # noqa: F841
# line 363
vr_start = 0.0
```

**Impact**: Low. The Lee model (Lee & Saw 2008) also treats the radial phase as starting from rest. The axial kinetic energy is typically <5% of the stored energy at phase transition. This is a known simplification, not a bug.

---

## LOW Findings

### LOW-1: `verify_force_balance` uses analytical dp/dr only

- **File**: `bennett_equilibrium.py:165`
- **Found by**: py-circuit-val (V-BEN-1)
- **Cross-review verdict**: **CONFIRMED** — the function name is slightly misleading

The function verifies that `dp/dr_analytical + J_z * B_theta = 0`, which tests the self-consistency of the analytical expressions. It does NOT verify a numerically-initialized state against the equilibrium condition. The function name `verify_force_balance` is accurate (it does verify force balance — of the analytical profiles), but a user might expect it to verify a numerical state.

### LOW-2: Temperature convention undocumented in `create_bennett_state`

- **File**: `bennett_equilibrium.py:179-242`
- **Found by**: py-circuit-val (V-BEN-2)
- **Cross-review verdict**: **CONFIRMED** — minor documentation gap

`Te` and `Ti` arrays contain temperatures in **Kelvin**, not eV. The Bennett relation uses `k_B * T`, so this is internally consistent. But callers should know the convention.

### LOW-3: `validate_full` mixes log-scale and linear-scale scoring

- **File**: `suite.py:520, 529-537`
- **Found by**: py-circuit-val (V-SUI-2)
- **Cross-review verdict**: **CONFIRMED** — design choice, not a bug

Neutron yield uses `log10(Y_sim/Y_exp)` as the "relative error", while circuit metrics use linear relative error. When both contribute to the overall score via `1 - error`, the scoring is not on comparable scales. However, the physics reviewer notes this is deliberate: neutron yield has order-of-magnitude shot-to-shot variability, so log-scale comparison is more appropriate. The issue is that the combined `validate_full` score mixes the two scales.

### LOW-4: `config_hash` uses `json.dumps(default=str)` — non-deterministic across Python versions

- **File**: `suite.py:244-246`
- **Found by**: py-circuit-val (V-SUI-3)
- **Cross-review verdict**: **CONFIRMED** — cosmetic

If config contains non-serializable objects, `default=str` converts them to strings, which may differ across Python versions. The hash is only used for tracking purposes, not for correctness.

---

## REJECTED Findings

### REJ-1: py-circuit-val V-LEE-1 — "Radial mass doesn't account for axial sweep"

The Python reviewer claimed the radial phase mass `M_slug = fm * rho0 * pi * (b^2 - r_s^2) * L_pinch` double-counts mass already swept during the axial phase. **REJECTED**: This is incorrect. The axial phase sweeps gas along the z-axis between the electrodes. The radial phase sweeps gas radially between the cathode (`r = b`) and the shock position (`r_s`), in the END region beyond the anode tip. These are geometrically disjoint volumes. The Lee model (Lee & Saw 2008, Section 2.2) uses the same `fm` for both phases by convention. The reviewer's claim that Lee uses "a separate mass fraction `f_mr` for the radial phase" is correct for some Lee model variants, but the standard published form uses a single `fm` for both phases.

The physics reviewer did NOT raise this as a bug, which supports the rejection.

### REJ-2: py-circuit-val CC-3 — No input validation on physical parameters

The reviewer suggested that functions like `bennett_density(r, n_0=-1e20, a=0.01)` should validate that physical parameters are positive. **REJECTED**: These are internal physics functions, not user-facing APIs. Adding parameter validation to every function would add significant overhead and code bulk without preventing any realistic misuse. The Pydantic config layer already validates user inputs.

### REJ-3: py-circuit-val V-NOH-2 — `B_0 != 0.0` exact float comparison

The reviewer flagged `B_0 != 0.0` in `magnetized_noh.py:212` as fragile. **REJECTED**: `B_0` is a user-provided parameter, not the result of computation. Comparing a user parameter against the exact value `0.0` (which is representable in IEEE 754) is safe. Users who want the unmagnetized case pass `B_0=0.0` or `B_0=0` — both produce exact zero.

---

## Positive Observations

1. **Bennett equilibrium (Grade A)**: Research-quality. All 8 functions analytically correct. Force balance residuals < 1e-12 (machine precision). References: Bennett (1934), Haines (2011).

2. **Magnetized Noh (Grade A)**: Full Velikovich et al. (2012) solution. Compression ratio via brentq, R-H residuals < 1e-10. Correctly handles cylindrical geometry factors (X^2 for density/B from upstream convergence).

3. **Experimental data curation**: PF-1000, NX2, UNU-ICTP parameters match published sources. 26-point digitized PF-1000 waveform with uncertainty budgets. GUM-style combined uncertainty computation.

4. **Cross-validation framework**: Methodologically sound train/test split pattern for parameter transferability testing.

5. **First-peak finder**: Correctly handles DPF-specific pattern where post-pinch oscillations can exceed the first peak amplitude.
