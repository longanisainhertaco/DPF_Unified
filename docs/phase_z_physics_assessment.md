# Phase Z Physics Assessment: Roadmap to 7.0/10

**Date**: 2026-02-25
**Author**: physics-dev agent
**Context**: Post-Phase Y analysis — identifying specific gaps blocking composite score of 7.0/10

---

## Executive Summary

Phase Y (100 new tests) addressed all major items from the PhD Debate #7 roadmap to 7.0. Analysis shows we
are now at an estimated **6.8/10** (up from 6.5), with specific remaining gaps preventing 7.0. The
primary blockers are: (1) neutron yield prediction not yet validated, (2) B-field initialization from
snowplow position in MHD grid incomplete, and (3) calibration convergence not benchmarked against
published Lee & Saw (2014) values.

---

## Current State After Phase Y

### What Phase Y Implemented

| Feature | Module | Test File | Status |
|---------|--------|-----------|--------|
| Reflected shock (Lee Phase 5) | `fluid/snowplow.py` | `test_phase_y_reflected_shock.py` (12 tests) | ✅ |
| m=0 sausage instability growth rate | `diagnostics/instability.py` | `test_phase_y_instability.py` | ✅ |
| Snowplow-MHD dynamic pressure coupling | `engine.py` | `test_phase_y_mhd_coupling.py` | ✅ |
| Digitized PF-1000 waveform (26 points) | `validation/experimental.py` | `test_phase_y_waveform.py` | ✅ |
| Cross-validation framework | `validation/calibration.py` | `test_phase_y_crossval.py` | ✅ |
| 3-term calibration objective (1+ DOF) | `validation/calibration.py` | `test_phase_y_waveform.py` | ✅ |
| `round()` fix for zipper BC index | `engine.py` | `test_phase_y_beam_target.py` | ✅ |
| Beam-target yield rate | `diagnostics/beam_target.py` | `test_phase_y_beam_target.py` | ✅ |

### Score Estimate After Phase Y

Using the PhD Debate #7 weighting formula:

| Component | Debate #7 | Est. After Y | Δ | Weight | Δ Contribution |
|-----------|-----------|--------------|---|--------|----------------|
| MHD Numerics | 7.75 | 7.75 | 0 | 20% | 0 |
| Transport | 7.5 | 7.5 | 0 | 15% | 0 |
| Circuit | 7.0 | 7.0 | 0 | 15% | 0 |
| **DPF-Specific** | **6.3** | **~7.2** | **+0.9** | **25%** | **+0.225** |
| **Validation** | **4.0** | **~5.3** | **+1.3** | **15%** | **+0.195** |
| AI/ML | 3.0 | 3.0 | 0 | 5% | 0 |
| Software Eng. | 7.5 | 7.5 | 0 | 5% | 0 |
| **Composite** | **6.44** | **~6.86** | **+0.42** | | |

**Estimated post-Phase Y score: ~6.8/10**

DPF-Specific improvement breakdown:
- Reflected shock (Lee Phase 5): +0.5 — completed item from required list
- m=0 growth rate diagnostic: +0.2 — onset detection implemented
- Two-way pressure coupling: +0.2 — MHD pressure feeds back to snowplow force

Validation improvement breakdown:
- Digitized PF-1000 waveform: +0.5 — enables 26-point NRMSE
- 3-term objective (1 DOF): +0.4 — addresses the 0-DOF criticism
- Cross-validation framework: +0.4 — calibrate on PF-1000, predict NX2

---

## Specific Gaps Blocking 7.0

### Gap 1: Neutron Yield Prediction Not Validated (HIGH PRIORITY)
**Current state**: `beam_target.py` implements `beam_target_yield_rate()`, but there is no comparison against
published experimental neutron yields. PhD Debate #7 identified this as a key validation milestone.

**What's needed**:
- PF-1000 published neutron yield: ~10^10 n/shot (deuterium, S. Lee & S.H. Saw 2008)
- NX2 published yield: ~10^8 n/shot at 400 kA
- Order-of-magnitude agreement (within factor 10) is the target

**Score impact**: Validation 5.3 → 6.0 (+0.7), Total composite +0.105

**Blocking condition**: `beam_target_yield_rate()` returns a rate [n/s], but integration over pinch dwell
time requires snowplow `pinch_time` and MHD-extracted density/temperature at pinch — both available.

---

### Gap 2: Calibration Not Benchmarked Against Published fc/fm (MEDIUM PRIORITY)
**Current state**: `LeeModelCalibrator` optimizes fc/fm but doesn't compare results against Lee & Saw
(2014) Table 1 published values (fc ≈ 0.7, fm ≈ 0.05-0.15 for PF-1000).

**What's needed**:
- Run `calibrate_default_params("PF-1000")` with maxiter=200
- Compare result fc/fm to published: fc ∈ [0.65, 0.80], fm ∈ [0.05, 0.20]
- Add test verifying calibrated fc/fm falls within published ranges

**Score impact**: Validation 5.3 → 5.6 (+0.3), Total composite +0.045

**Engineering note**: If calibrated fc/fm is wildly different from published (e.g., fc < 0.5 or fm > 0.4),
this indicates the Lee model implementation has a physics error — high priority to catch.

---

### Gap 3: B-field Initialization from Snowplow Position (MEDIUM PRIORITY)
**Current state**: Phase X added the radial zipper BC (zeroing B_theta outside r_shock). But Phase Y's
pressure coupling is one-directional: MHD grid pressure → snowplow force. The reverse coupling —
snowplow position (r_shock) initializing B_theta profile in MHD grid — is NOT implemented. The B_theta
field evolves independently from the snowplow dynamics.

**What's needed**:
- When entering radial phase: initialize B_theta(r) = μ₀I/(2πr) for r < r_shock, 0 otherwise
- Update MHD B_theta each step to maintain consistency with snowplow sheath position
- This closes the two-way coupling loop (pressure feedback already done in Y; B-field initialization missing)

**Score impact**: DPF-Specific 7.2 → 7.5 (+0.3), Total composite +0.075

**Implementation risk**: Medium — modifying B_theta in the MHD grid during time evolution could
destabilize the MHD solve. Need to operator-split this as a boundary condition update, not a source term.

---

### Gap 4: Cross-Validation Not Actually Run (LOW PRIORITY)
**Current state**: `CrossValidator.validate("PF-1000", "NX2")` is implemented and tested with mocks, but
there's no verification that cross-validation actually produces physically reasonable generalization scores
with real Lee model runs.

**What's needed**:
- Run CrossValidator integration test (currently marked `@pytest.mark.slow`)
- Verify generalization_score > 0.5 (i.e., PF-1000 fc/fm predicts NX2 within 50% errors)
- If score < 0.5, investigate why (geometry mismatch, parameter insensitivity, etc.)

**Score impact**: Validation 5.3 → 5.5 (+0.2) — infrastructure credit already given; full credit on pass

---

### Gap 5: Waveform Quality and Coverage (LOW PRIORITY)
**Current state**: PF-1000 waveform has 26 digitized points. NX2 has no waveform. The 26-point
resolution provides ~24 independent values (25 intervals minus 1 for normalization) for the 2-parameter
calibration — adequate but not rich.

**What's needed**:
- Verify 26-point PF-1000 waveform accurately represents Scholz (2006) Fig. 3 discharge shape
  - Key: current dip at ~5.8 μs and zero-crossing at ~11.6 μs should be captured
- Add NX2 waveform from Bernard & Sadowski (2006) or equivalent reference: ~10 points minimum
- Tests exist and pass; this is about scientific accuracy of the digitization

**Score impact**: Validation 5.3 → 5.6 (+0.3) — marginal improvement, low priority

---

## Implementation Plan: Phase Z Work Items

### Z.1 — Neutron Yield Validation (CRITICAL PATH for 7.0)

**Target**: `src/dpf/validation/neutron_validation.py` (new file or extension of `experimental.py`)

**Algorithm**:
1. Extract pinch parameters from SnowplowModel at `_pinch_complete=True`:
   - r_pinch_min (≈ 0.1*a)
   - pinch duration (≈ r_pinch_min / v_Alfven ≈ 10-100 ns)
   - swept mass → density at pinch via volume: n_e = M_slug / (m_D * π * r_pinch_min² * L_pinch)
   - Temperature from adiabatic compression or Lee model T_e formula
2. Call `beam_target_yield_rate(n_e, T_e, volume)` × pinch_duration → total neutrons
3. Compare to published values with factor-of-10 tolerance

**Reference values** (for test assertions):
| Device | I_peak | Neutron yield | Source |
|--------|--------|---------------|--------|
| PF-1000 | 1.87 MA | ~10^10 n/shot | Lee & Saw, J. Fusion Energy (2008) |
| NX2 | 400 kA | ~10^8 n/shot | Saw et al., AIP Conf. Proc. (2010) |
| UNU-ICTP | 30 kA | ~10^6 n/shot | Lee model reference values |

**Tests to write**: `tests/test_phase_z_neutron_yield.py` (15-20 tests)
- Order-of-magnitude agreement for PF-1000 (within factor 10)
- Scaling: yield ∝ I^4 for low-yield machines (Roth regression)
- NaN-free for all physically valid inputs
- Yield = 0 when pinch not complete

---

### Z.2 — Calibration Benchmark Against Published fc/fm

**Target**: Extension of `LeeModelCalibrator` + new test class

**Algorithm**:
1. Add `LeeModelCalibrator.benchmark_against_published()` method:
   - Runs calibration with maxiter=200
   - Compares fc/fm to hardcoded literature ranges per device
   - Returns `{"in_range": bool, "fc_published": (0.65, 0.80), "fm_published": (0.05, 0.20)}`
2. Add `published_fc_range` and `published_fm_range` to `ExperimentalDevice` dataclass

**Literature values** (Lee & Saw 2014, Table 1):
| Device | fc range | fm range |
|--------|----------|----------|
| PF-1000 | 0.65-0.75 | 0.05-0.15 |
| NX2 | 0.60-0.80 | 0.10-0.20 |
| UNU-ICTP | 0.55-0.75 | 0.15-0.30 |

**Tests**: `tests/test_phase_z_calibration_benchmark.py` (10 tests)
- fc within published range for PF-1000 (slow integration test)
- fm within published range for PF-1000 (slow integration test)
- Convergence flag is True after maxiter=200
- `benchmark_against_published()` returns dict with correct keys

---

### Z.3 — B-field Initialization from Snowplow (DEFERRED to Z.2+)

**Risk**: High coupling complexity. Defer until neutron yield (Z.1) and calibration (Z.2) are verified.

**Design sketch**:
1. `_apply_snowplow_bfield_ic(engine)` — called when snowplow enters radial phase:
   - Sets `state["B"][1, :ir_shock, :, :]` = `mu_0 * I / (2*pi * r_grid[:ir_shock])`
   - Sets `state["B"][1, ir_shock:, :, :]` = 0 (thin-sheath)
2. Called once at radial phase entry, then zipper BC maintains the boundary each step
3. Net effect: MHD B_theta starts from Lee model prediction, not zero

**Implementation file**: `src/dpf/engine.py` — new method `_initialize_radial_bfield()`

---

### Z.4 — Minor Improvements (low engineering cost)

1. **Fix ir_shock index clamping**: Verify `round()` fix from Phase Y is correctly applied in all
   relevant locations in `engine.py` (zipper BC for both axial and radial phases).

2. **LHDI v_d diagnostic logging**: Per Dr. PP's Debate #7 suggestion — log `v_d/v_threshold` ratio at
   each timestep as a debug diagnostic. Confirms LHDI is triggered throughout discharge.

3. **Calibration convergence diagnostics**: Add `n_evals` and `success` to log output. If
   `success=False` with default maxiter=100, increase default to 200.

---

## Priority Order

| Priority | Item | Score Δ | Effort | Status |
|----------|------|---------|--------|--------|
| 1 | Z.1: Neutron yield validation | +0.105 | 2-3 hours | Not started |
| 2 | Z.2: Calibration benchmark | +0.045 | 1-2 hours | Not started |
| 3 | Run Phase Y slow tests | +0.02 | 30 min | Not started |
| 4 | Z.3: B-field init from snowplow | +0.075 | 4-6 hours | Deferred |
| 5 | Z.4: Minor fixes | +0.01 | 30 min | Not started |

**Combined Phase Z potential**: 6.8 + 0.105 + 0.045 + 0.075 = **~7.0/10** ✓

The roadmap to 7.0 is achievable with Z.1 + Z.2 + Z.3. The bottleneck is Z.3 (B-field coupling
complexity). If Z.3 is deferred, Z.1 + Z.2 alone may only reach ~6.95 — close but possibly not enough
depending on panel weighting.

**Conservative estimate**: Z.1 + Z.2 → **6.9/10**
**Full estimate with Z.3**: Z.1 + Z.2 + Z.3 → **7.0-7.1/10**

---

## O(L)² / QSSS Integration (BLOCKED on Task #2 Research)

Per team instructions, O(L)² and QSSS-related changes are blocked until research agent (task #2)
delivers findings. Expected impact on scoring:

- O(L)² could improve **Circuit** component: 7.0 → 7.5 (+0.075 composite)
- QSSS could improve **MHD Numerics**: 7.75 → 8.25 (+0.100 composite)
- Combined: 6.9 → **7.1/10** (or 7.2 if Z.3 also complete)

---

## Key Physics References

1. **Lee & Saw (2014)**: fc = 0.7, fm = 0.05-0.15 for PF-1000 calibration benchmarks
   - doi:10.1007/s10894-014-9756-4
2. **Scholz (2006)**: PF-1000 I(t) waveform, I_peak = 1.87 MA at ~5.8 μs
   - 10th Int. Conf. Dense Z-Pinches
3. **Davidson & Gladd (1975)**: LHDI threshold formula, v_d > (m_e/m_i)^{1/4} × v_ti
   - Phys. Fluids 18:1327
4. **Haines (2011)**: m=0 sausage instability growth rate formula
   - doi:10.1088/0741-3335/53/9/093001
5. **Lee & Saw (2008)**: Neutron yield scaling, D-D reaction rate
   - J. Fusion Energy 27:292-295

---

*Assessment prepared for team-lead review. Waiting on task #2 (O(L)² research) before implementing
Circuit/MHD improvements. Phase Z implementation can begin immediately with Z.1 and Z.2.*
