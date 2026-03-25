# Sprint 6 Research: MLX Calibration Sweep & GPU Optimization Round 2

**Date**: 2026-03-24
**Author**: dpf-mhd-physicist (Cortana)
**Status**: Research complete -- ready for implementation
**Methodology**: Six Sigma DMAIC
**Scope**: fc/fm calibration for MLX backend, GPU performance optimization

---

## Table of Contents

1. [Item 1: fc/fm Calibration Sweep for MLX Backend](#item-1-fcfm-calibration-sweep-for-mlx-backend)
   - [1.1 Define](#11-define)
   - [1.2 Measure](#12-measure)
   - [1.3 Analyze](#13-analyze)
   - [1.4 Improve](#14-improve)
   - [1.5 Control](#15-control)
2. [Item 2: GPU Optimization Round 2](#item-2-gpu-optimization-round-2)
   - [2.1 Define](#21-define)
   - [2.2 Measure](#22-measure)
   - [2.3 Analyze](#23-analyze)
   - [2.4 Improve](#24-improve)
   - [2.5 Control](#25-control)
3. [Risk Register](#risk-register)
4. [References](#references)

---

## Item 1: fc/fm Calibration Sweep for MLX Backend

### 1.1 Define

#### Problem Statement

The MLX MHD solver (WENO5-Z + HLLD + SSP-RK3 in float32) has no calibrated fc/fm parameters. The existing `pf1000` and `pf1000_akel` presets carry parameters tuned against the Python Lee-model ODE solver, not the MLX finite-volume MHD solver. Running the MLX backend with these preset values will produce different I(t) waveforms because the numerical diffusion profile, coordinate handling, and unit system differ fundamentally from the lumped-circuit Lee model.

#### Goal

Produce a validated `pf1000_mlx` parameter set (or fc/fm correction factors for the existing presets) that achieves:
- **Primary**: NRMSE(I(t)) < 0.15 vs Scholz (2006) experimental PF-1000 waveform at 27 kV / 3.5 Torr D2
- **Secondary**: I_peak relative error < 5%
- **Tertiary**: Consistency with Akel (2021) 24-shot statistical validation (mean |I_peak error| < 5%)

#### Scope

- Parameter sweep over fc, fm grid
- MLX backend only (Python and Metal backends already calibrated)
- PF-1000 device at the Scholz (2006) operating point (27 kV, 3.5 Torr D2)
- Validation extension to Akel (2021) 16 kV multi-shot data after primary calibration

#### Out of Scope

- Radial-phase parameters (fmr, fcr) -- hold fixed at Lee & Saw (2014) published values
- Circuit parameter re-calibration (C, L0, R0) -- these are experimentally measured, not free parameters
- Multi-device calibration (NX2, UNU-ICTP) -- defer to Sprint 7

### 1.2 Measure

#### Current Calibration State

**Preset: `pf1000`** (`src/dpf/presets.py:65-115`)

| Parameter | Value | Source | Notes |
|-----------|-------|--------|-------|
| fc | 0.70 | Lee & Saw (2014) | Current fraction: fraction of total discharge current flowing through the current sheet |
| fm | 0.08 | Lee & Saw (2014) | Mass fraction: fraction of inter-electrode fill gas swept up by the current sheet |
| fmr | 0.16 | Lee & Saw (2014) | Radial mass fraction |
| C | 1.332 mF | Scholz (2006) Table 1 | Measured |
| V0 | 27 kV | Scholz (2006) | Operating voltage |
| L0 | 33.5 nH | Scholz (2006) | External inductance from short-circuit calibration |
| R0 | 2.3 mOhm | Scholz (2006) | External resistance |
| anode_radius | 115 mm | Scholz (2006) | |
| cathode_radius | 160 mm | Lee & Saw (2014) | Effective value |

**Preset: `pf1000_akel`** (`src/dpf/presets.py:116-172`)

| Parameter | Value | Source | Notes |
|-----------|-------|--------|-------|
| fc | 0.70 | Akel (2021), all 24 shots | Held constant across all shots |
| fm | 0.19 | Akel (2021) average | Overridden per-shot (range 0.17--0.24) |
| R0 | 8.73 mOhm | 2.3 + 6.43 correction | EMPIRICAL: 6.43 mOhm offset calibrated 2026-03-15 |
| V0 | 27 kV | Akel (2021) | 16 kV operating point in paper, but V0=27kV in preset |

**Validation Baseline (Python Lee model + `pf1000_akel`)**:
- I_peak error: 4.1% (single reference shot)
- NRMSE: 0.146
- Validated against: Scholz (2006) experimental I(t) waveform

#### What fc and fm Physically Represent

The Lee model (Lee & Saw 2014) introduces two phenomenological coupling factors between the lumped-circuit model and the plasma dynamics:

**fc (current fraction, 0 < fc <= 1)**: The fraction of total discharge current that actually flows through the moving current sheet. The remainder (1-fc) leaks through the insulator surface, electrode-gap sparking, or pre-ionization channels that bypass the main sheet. In the snowplow ODE, the magnetic driving force scales as (fc*I)^2. For PF-1000: fc = 0.70 means 30% current leakage, consistent with the large cathode array (12 rods) providing multiple alternative current paths.

**fm (mass fraction, 0 < fm <= 1)**: The fraction of fill gas between the electrodes that is actually swept up and accelerated by the current sheet. The remainder is left behind (gas leakage through gaps in the sheet, incomplete snowplow). In the ODE, the inertial mass scales as fm * rho0 * A * z. For PF-1000 at 27 kV / 3.5 Torr: fm = 0.08 is remarkably low, meaning 92% of the fill gas is NOT swept. This is characteristic of MJ-class devices where the sheath velocity (Va ~ 10 cm/us) far exceeds the sound speed, creating a thin shell that leaves most gas undisturbed.

**Critical insight**: In the MHD solver, fc and fm do NOT appear as explicit parameters. Instead, they are encoded implicitly through:
1. **Initial conditions**: rho0 = fill density (fm determines effective inertia)
2. **Electrode boundary conditions**: B_theta = mu0 * fc * I / (2*pi*r) at the cathode ghost cells
3. **Numerical diffusion**: WENO5-Z + HLLD has different effective dissipation than the Lee model's ODE solver

The mapping from Lee-model (fc, fm) to MHD-solver initial/boundary conditions is:
- `rho_eff = fm * rho_fill` in the initial density field (or equivalently, scale fill pressure)
- `I_eff = fc * I_circuit` in the electrode BC

#### How the Existing Calibration Pipeline Works

The validation pipeline (`src/dpf/validation/lee_model_comparison.py`) implements:

1. **LeeModel class** (line 184): Integrates coupled circuit + snowplow ODEs using `scipy.integrate.solve_ivp`
2. **Phase 1 (axial rundown)**: Snowplow equation d^2z/dt^2 = (mu0/4pi) * ln(b/a) * (fc*I)^2 / M_swept
3. **Phase 2 (radial implosion)**: Slug model d^2r_s/dt^2 = -(mu0/4pi) * I^2 / (rho * r_s * L_pinch)
4. **Device parameter lookup**: From `dpf.validation.experimental.DEVICES` registry
5. **Lee-specific fc/fm**: Device registry can carry `lee_fc`, `lee_fm` that override constructor defaults
6. **Crowbar model**: Voltage-zero trigger, L-R decay post-crowbar
7. **Comparison metrics**: `LeeModelComparison` dataclass with peak_current_error, waveform_nrmse

The pipeline does NOT currently support MLX backend runs -- it compares Lee-model ODE output to experimental data. For Sprint 6, we need a new pipeline that:
1. Runs the MLX solver with given (fc, fm) encoded in initial/boundary conditions
2. Extracts I(t) from the solver's circuit coupling state
3. Compares against the same Scholz (2006) experimental waveform

### 1.3 Analyze

#### Root Cause: Why MLX Needs Separate Calibration

Five sources of systematic difference between the Lee-model calibration and MLX solver behavior:

**1. Numerical Diffusion (WENO5-Z vs ODE)**

The Lee model solves a 4-variable ODE system with `solve_ivp` (adaptive RK45, effectively infinite spatial resolution). The MHD solver discretizes the full Euler/MHD system on a (240, 1, 800) grid with:
- WENO5-Z reconstruction: 5th-order in smooth regions, drops to ~2nd at discontinuities (sheath front)
- HLLD Riemann solver: exact resolution of 7 MHD waves, but introduces numerical dissipation at each interface
- SSP-RK3: 3rd-order temporal, CFL-limited dt ~ 1e-10 s

The effective numerical viscosity broadens the sheath, changes its velocity, and modifies the current dip timing. This systematically alters the I(t) waveform shape.

**2. Unit System (HL vs SI)**

The MLX solver operates in Heaviside-Lorentz units (mu0 = 1) internally. B-field conversion: B_HL = B_SI / sqrt(mu0). The `convert_b_si_to_hl` flag in MLXMHDSolver handles this, but any mismatch produces magnetic forces that are off by factors of mu0 (= 1.26e-6). MEMORY.md documents this exact bug: "SI->HL B-field conversion: Metal solver uses HL units (mu0=1). Input B in Tesla must be divided by sqrt(mu0). Without this, magnetic forces are 10^6x too weak."

**3. Coordinate Handling (Cylindrical Finite-Volume)**

The MLX solver uses r-weighted finite-volume differencing:
```
dU/dt = -(1/(r*dr)) * (r_{i+1/2}*F_r - r_{i-1/2}*F_r) - (F_z - F_z)/dz + S_geom
```
with geometric source terms S_mr = (rho*vt^2 - Bt^2)/r and S_mt = -2*(rho*vr*vt - Br*Bt)/r. The Lee model has no radial structure -- it assumes a thin shell. Resolving the radial profile changes the effective inertia and inductance.

**4. Float32 Precision**

MLX operates in float32 (Metal has no float64). The dual-energy entropy tracer mitigates catastrophic cancellation in pressure recovery, but accumulated round-off in the WENO5-Z smoothness indicators (beta0, beta1, beta2) and HLLD intermediate states affects the solution. The `eps=1e-6` floor in WENO-Z weights (CLAUDE.md lesson: "WENO5-Z eps must be 1e-6 (not 1e-36) in float32") prevents underflow but adds O(1e-6) noise to the weights.

**5. Compensating Error Legacy**

MEMORY.md explicitly warns: "Compensating errors: parameters calibrated with a bug compensate for it. Fixing the bug without recalibrating makes results WORSE." The existing fc=0.70, fm=0.08 values were calibrated against the Lee-model ODE, which has its own simplifications (no radial structure, slug model for implosion, no MHD waves). These compensating errors are NOT transferable to the MHD solver.

#### Parameter Sensitivity Analysis

From MEMORY.md: "Yn dominated by fill pressure (ST=0.90 Sobol). fc/fm barely matter (ST<0.08)." However, this Sobol analysis was for neutron yield, not I(t) waveform. For the current waveform:

- **fm dominates I(t) shape during axial phase**: Higher fm means more swept mass, slower sheath, later current peak. The snowplow force F ~ I^2 / M_swept, so M_swept = fm * rho0 * A * z directly controls sheath dynamics.
- **fc dominates I_peak magnitude**: I_peak scales roughly as fc * I_sc * exp(-R*T/4 / 2L), where I_sc = V0 * sqrt(C/L0) is the short-circuit peak. Reducing fc from 0.70 to 0.60 reduces I_peak by ~14%.
- **Parameter degeneracy**: "I_peak alone doesn't constrain (fc, fm) -- need NRMSE or Yn." Multiple (fc, fm) pairs can give the same I_peak with different waveform shapes.

#### Akel (2021) Reference Data Summary

From the extracted paper data (`akel-2021-pf1000.md`):

- **24 shots** at 16 kV, deuterium, 1.05 and 1.20 Torr
- fc = 0.70 (constant all shots)
- fm = 0.17--0.24 (varies by shot, avg ~0.20)
- r0 = 4.0--6.5 mOhm (varies by shot -- spark gap resistance)
- I_peak = 1131--1335 kA
- L0 = 25.0 nH, C = 1332 uF

Note the key difference: Akel uses **16 kV** (not 27 kV like the Scholz preset), and fm values are 0.17--0.24 (much higher than Scholz preset's fm=0.08). This is because fm depends on fill pressure and operating point -- not just geometry.

### 1.4 Improve

#### Parameter Space Definition

Based on the analysis, the sweep covers:

**fc sweep**: [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
- 10 values
- Physical range: 0.50 (high leakage, typical of early shots) to 0.95 (nearly ideal sheet)
- Published PF-1000 values cluster at 0.70 (Lee & Saw 2014, Akel 2021)

**fm sweep**: [0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30]
- 14 values
- Physical range: 0.04 (very thin sheet, MJ-class fast device) to 0.30 (thick sheet, moderate pressure)
- Scholz (27 kV / 3.5 Torr): fm=0.08; Akel (16 kV / 1.2 Torr): fm=0.17--0.24

**Total grid points**: 10 x 14 = 140 simulations

#### Objective Function

```python
def objective(fc: float, fm: float, I_exp_t: np.ndarray, I_exp: np.ndarray) -> dict:
    """Run MLX solver and compute metrics against experimental I(t).

    Args:
        fc: Current fraction [0.5, 0.95].
        fm: Mass fraction [0.04, 0.30].
        I_exp_t: Experimental time array [s].
        I_exp: Experimental current array [A].

    Returns:
        dict with keys: nrmse, ipeak_error, ipeak_sim, timing_error
    """
    # 1. Encode fc into electrode BC: I_eff = fc * I_circuit
    # 2. Encode fm into initial density: rho0_eff = fm * rho_fill
    #    (or scale fill_pressure_Pa by fm relative to nominal)
    # 3. Run MLX solver for sim_time = 10 us
    # 4. Extract I(t) from coupling_interface
    # 5. Interpolate to experimental time grid
    # 6. Compute NRMSE = ||I_sim - I_exp|| / ||I_exp||
    # 7. Compute I_peak error = |I_peak_sim - I_peak_exp| / I_peak_exp
    nrmse = np.sqrt(np.mean((I_sim_interp - I_exp)**2)) / np.sqrt(np.mean(I_exp**2))
    ipeak_error = abs(I_sim.max() - I_exp.max()) / I_exp.max()
    return {"nrmse": nrmse, "ipeak_error": ipeak_error}
```

**Primary objective**: minimize NRMSE
**Constraint**: ipeak_error < 0.05

#### Sweep Procedure

```python
# Pseudocode for calibration sweep
import itertools
import numpy as np

fc_vals = np.arange(0.50, 0.96, 0.05)  # 10 values
fm_vals = np.arange(0.04, 0.31, 0.02)  # 14 values

results = {}
for fc, fm in itertools.product(fc_vals, fm_vals):
    # Modify preset in-memory (do NOT write to presets.py)
    config = get_preset("pf1000")
    config["snowplow"]["current_fraction"] = fc
    config["snowplow"]["mass_fraction"] = fm

    # Run MLX solver
    solver = MLXMHDSolver(
        grid_shape=config["grid_shape"],
        dx=config["dx"],
        gamma=5/3,
        reconstruction="weno5z",
        riemann_solver="hlld",
        time_integrator="ssp_rk3",
        convert_b_si_to_hl=True,
    )
    # Initialize from config, run for sim_time
    # ... (engine wiring)

    metrics = objective(fc, fm, I_exp_t, I_exp)
    results[(fc, fm)] = metrics

# Find optimal
best = min(results, key=lambda k: results[k]["nrmse"])
```

#### Runtime Estimate (M3 Pro)

- MLX solver step time: ~2 ms per step at (240, 1, 800) grid (estimated from Sprint 4 benchmarks)
- CFL dt ~ 1e-10 s, sim_time = 10e-6 s --> ~100,000 steps per simulation
- Time per simulation: ~200 s (3.3 min)
- 140 simulations: ~28,000 s = **7.8 hours** sequential

**Optimization strategies**:
1. **Coarse-to-fine**: Run 5x7=35 point coarse grid first (fc step 0.10, fm step 0.04), identify promising region, then refine with 0.02 steps. Cost: ~35 * 200s + ~36 * 200s = ~14,200s = **3.9 hours**
2. **Reduced resolution**: Use (120, 1, 400) grid for initial sweep -- 4x fewer cells, ~4x faster per step. Then validate winners at full (240, 1, 800).
3. **Early termination**: If NRMSE > 0.50 by t = 3 us (half the rise time), abort that (fc, fm) pair.
4. **Parallel**: MLX uses Metal GPU; can run 2-3 simulations concurrently if memory permits (each ~800MB VRAM at full resolution).

**Recommended approach**: Coarse-to-fine at reduced resolution, validate top-5 at full resolution.
- Estimated total runtime: **2-3 hours** on M3 Pro.

#### Detecting Compensating Errors

Compensating errors occur when wrong physics + wrong parameters accidentally produce right answers. Detection:

1. **Multi-metric validation**: Require BOTH I(t) waveform NRMSE < 0.15 AND I_peak error < 5%. A pair that nails I_peak but has bad waveform shape is compensating.

2. **Cross-condition validation**: Calibrate at 27 kV / 3.5 Torr (Scholz), then validate at 16 kV / 1.2 Torr (Akel) WITHOUT re-tuning. If NRMSE degrades by > 2x, the calibration is overfitting to one condition.

3. **Physical plausibility check**: Optimal fc should be in [0.60, 0.85]. Optimal fm should be in [0.05, 0.20] for 27 kV / 3.5 Torr. Values outside these ranges indicate compensating errors masking a physics bug.

4. **Conservation audit**: At the optimal (fc, fm), verify:
   - Total energy conservation: |dE/E0| < 1e-6 per step
   - Mass conservation: |dM/M0| < 1e-8
   - div(B) < 1e-10

5. **Backend parity**: Compare MLX I(t) at optimal (fc, fm) against Python engine I(t) at published (fc, fm). If the waveform shapes differ qualitatively (different number of oscillations, different dip depth), the numerical diffusion difference is too large and indicates a bug, not just calibration offset.

### 1.5 Control

#### Deliverables

1. **Script**: `scripts/calibrate_mlx_pf1000.py` -- standalone sweep script with CLI args for fc/fm ranges, grid resolution, output directory
2. **Results**: `data/calibration/mlx_pf1000_sweep.json` -- all (fc, fm, NRMSE, ipeak_error) results
3. **Preset update**: Add `pf1000_mlx` preset to `src/dpf/presets.py` with calibrated values
4. **Test**: `tests/test_mlx_calibration.py` -- regression test that verifies the calibrated preset achieves NRMSE < 0.15

#### Monitoring

- Track NRMSE and ipeak_error in CI for the `pf1000_mlx` preset
- If any future physics change (reconstruction, Riemann solver, source terms) causes NRMSE regression > 0.02, re-run calibration sweep
- Document the calibration provenance: "pf1000_mlx: fc=X, fm=Y calibrated 2026-03-XX against Scholz (2006) 27 kV / 3.5 Torr with MLX WENO5-Z+HLLD+SSP-RK3 at (240,1,800) grid"

---

## Item 2: GPU Optimization Round 2

### 2.1 Define

#### Problem Statement

The MLX MHD solver achieves 2.68x speedup over PyTorch MPS (MEMORY.md), but profiling reveals three categories of GPU inefficiency:
1. **split+stack anti-pattern**: 4 call sites in `mlx_timestepper.py` and 1 in `mlx_solver.py` decompose the (10, nr, nz) state into 10 separate arrays, modify 1-3, then re-stack. Each split+stack is O(10 * nr * nz) memory traffic for O(1) logical modifications.
2. **`_take` index array allocation**: 17 calls per WENO5-Z reconstruction in `mlx_reconstruction.py`, each creating a fresh `mx.array(list(range(...)))` Python-side index array. Over 2 dimensions x 3 RK stages = 6 WENO5-Z calls per step, totaling 102 index array allocations per timestep.
3. **mx.compile() coverage gaps**: Only `cons_to_prim`, `recover_pressure`, and `weno5z_left_biased` are compiled. The geometric source terms, entropy resync, floor application, and velocity clamping are uncompiled.

#### Goal

- **Primary**: Reduce per-step wall-clock time by 20-30% at (240, 1, 800) resolution
- **Secondary**: Eliminate Python-side allocations from the hot path (zero GC pressure during RK stages)
- **Tertiary**: Maintain bit-exact numerical results (optimizations must NOT change physics)

#### Scope

- MLX solver only (`src/dpf/metal/mlx_*.py`)
- Correctness-preserving transformations (no algorithm changes)
- M3 Pro target hardware

### 2.2 Measure

#### Current mx.compile() Usage

| Function | File | Compiled? | Calls/Step | Notes |
|----------|------|-----------|------------|-------|
| `_cons_to_prim_impl` | mlx_primitives.py:69 | Yes | 6+ | Via lazy cache |
| `_recover_pressure_impl` | mlx_primitives.py:88 | Yes | 3 | Via lazy cache |
| `_weno5z_left_biased` | mlx_reconstruction.py:177 | Yes | 12 | 2 dims x 2 (L+R) x 3 stages |
| `_geometric_sources` | mlx_timestepper.py:96 | **No** | 3 | One per RK stage |
| `_apply_floors` | mlx_timestepper.py:208 | **No** | 4 | Pre-RHS + 3 stages |
| `_clamp_velocity` | mlx_timestepper.py:265 | **No** | 3 | One per RK stage |
| `_resync_energy` | mlx_timestepper.py:319 | **No** | 3 | One per RK stage (dual-energy) |
| `fast_magnetosonic` | mlx_primitives.py:227 | **No** | 6+ | CFL + velocity clamp |
| `_hll_flux` | mlx_riemann.py:87 | **No** | N/A | NumPy bridge, not compilable |

#### split+stack Pattern Analysis

The anti-pattern at `mlx_timestepper.py:230,306,341` and `mlx_solver.py:716`:

```python
rows = list(mx.split(U, NVAR, axis=0))  # 10 separate (1, nr, nz) arrays
rows[IMR] = modified_value[None]          # modify 1-3 rows
rows[IEN] = modified_energy[None]
return mx.stack([r[0] for r in rows], axis=0)  # re-combine
```

**Cost per call**: `mx.split` creates 10 lazy views (cheap), but `mx.stack` forces materialization of all 10 arrays into a contiguous buffer. The `[0]` indexing squeezes the leading dim, creating 10 temporary (nr, nz) arrays.

**Frequency**: Called 4 times per RK stage (_apply_floors, _clamp_velocity, _resync_energy) + 1 for CT = **13 split+stacks per timestep** with SSP-RK3.

At (240, 800) resolution, each split+stack touches 10 * 240 * 800 * 4 bytes = 7.68 MB. Total: 13 * 7.68 = **99.8 MB of unnecessary memory traffic per timestep**.

#### _take Index Array Analysis

`_take` in `mlx_reconstruction.py:38`:
```python
def _take(arr, axis, start, length):
    idx = mx.array(list(range(start, start + length)), dtype=mx.int32)
    return mx.take(arr, idx, axis=axis)
```

**Problem**: `list(range(start, start + length))` creates a Python list, then `mx.array()` copies it to Metal. For WENO5-Z at (240, 800):
- Per-dimension: 10 `_take` calls for L+R stencil extraction + 8 for PLM fallback path
- Per-step: 2 dimensions x 3 RK stages x 10 calls = **60 _take calls minimum**
- Each creates an int32 index array of length ~235 (240 - 5 = 235 interfaces)

The Python allocation is the bottleneck, not the mx.take itself (which is lazy).

#### Completed Optimizations (Sprint 5 baseline)

Already done:
- Geometric sources computed on-device (no CPU roundtrip)
- CFL computation on-device with single GPU->CPU transfer (2 floats)
- mx.eval barriers removed from intermediate stages
- Vacuum cell masking in CFL (prevents frozen dt)

### 2.3 Analyze

#### Root Cause Ranking

| # | Issue | Impact | Effort | Risk |
|---|-------|--------|--------|------|
| 1 | split+stack anti-pattern (13x/step) | High (100 MB/step traffic) | Medium (refactor 5 functions) | Low -- pure refactor |
| 2 | _take index allocation (60+/step) | Medium (Python GC pressure, launch overhead) | Low (cache indices) | None -- pure optimization |
| 3 | Uncompiled elementwise chains | Medium (missed fusion opportunities) | Low (wrap in compile) | Low -- verify bit-exact |
| 4 | HLL NumPy bridge in mlx_riemann.py | High for HLL users (CPU roundtrip) | High (rewrite in MLX) | Medium -- HLLD already native |
| 5 | Redundant cons_to_prim calls | Low (already compiled, lazy eval) | Low | None |

#### Alternative Patterns to split+stack

**Option A: Direct array indexing with mx.where**
```python
# Instead of split+modify+stack, use masked assignment
mask_mr = mx.array([0,0,0,1,0,0,0,0,0,0], dtype=mx.bool_)[:, None, None]
U_new = mx.where(mask_mr, new_mr_value[None], U)
```
Problem: Requires one mx.where per modified component. For 3 modifications, that's 3 full-array reads+writes.

**Option B: Concatenate slices (zero-copy views)**
```python
# Replace rows 3 and 4 of a (10, nr, nz) array:
U_new = mx.concatenate([
    U[:3],           # rows 0-2 (view, zero-copy)
    new_row3[None],  # row 3
    new_row4[None],  # row 4
    U[5:],           # rows 5-9 (view, zero-copy)
], axis=0)
```
This is the most natural pattern. MLX slicing `U[:3]` creates a view (no copy). Concatenation only copies the modified rows. Cost: modified_rows * nr * nz * 4 bytes instead of 10 * nr * nz * 4.

**Recommendation**: Option B. Reduces memory traffic by 7x when modifying 3 of 10 components.

**Option C: In-place mutation (if MLX supports it)**
MLX arrays are immutable by design (functional semantics for automatic differentiation). No in-place mutation is possible. Option B is the best available pattern.

#### _take Optimization: Cached Index Arrays

The index arrays only depend on (start, length), not on the data. Cache them:

```python
_INDEX_CACHE: dict[tuple[int, int], mx.array] = {}

def _take(arr: mx.array, axis: int, start: int, length: int) -> mx.array:
    key = (start, length)
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = mx.array(list(range(start, start + length)), dtype=mx.int32)
    return mx.take(arr, _INDEX_CACHE[key], axis=axis)
```

Even better: use slicing instead of mx.take where possible:
```python
# _take(Q, axis=1, start=2, n_iface) is equivalent to:
Q[:, 2:2+n_iface, :]  # for axis=1
```

MLX slicing returns views (zero-copy). This eliminates both the Python allocation AND the mx.take gather kernel.

**Recommendation**: Replace _take with direct slicing. Falls back to mx.take only for non-contiguous access patterns (none exist in the current code).

#### mx.compile() Expansion

Functions safe to compile (pure elementwise, no Python control flow):
1. `_geometric_sources` -- pure arithmetic on U components
2. `fast_magnetosonic` -- pure arithmetic with clamps
3. `_apply_floors` -- needs refactoring to eliminate split+stack first
4. `_clamp_velocity` -- needs refactoring to eliminate split+stack first
5. `_resync_energy` -- calls recover_pressure (already compiled) + arithmetic

Functions NOT safe to compile:
- `_hll_flux` -- NumPy bridge, not an MLX function
- `ssp_rk3_step` -- calls mhd_rhs which has Python control flow (method dispatch)
- `compute_dt_cfl` -- has CPU transfer (float() calls)

### 2.4 Improve

#### Ranked Optimization Plan

**Priority 1: Replace _take with slicing** (Est. speedup: 5-8%, effort: 1 hour)

Before:
```python
qm2 = _take(Q, axis, 0, n_iface)
qm1 = _take(Q, axis, 1, n_iface)
q0  = _take(Q, axis, 2, n_iface)
```

After (for axis=1, radial direction):
```python
qm2 = Q[:, 0:n_iface, :]
qm1 = Q[:, 1:1+n_iface, :]
q0  = Q[:, 2:2+n_iface, :]
```

For axis=2 (axial direction):
```python
qm2 = Q[:, :, 0:n_iface]
qm1 = Q[:, :, 1:1+n_iface]
q0  = Q[:, :, 2:2+n_iface]
```

Need a helper that dispatches on axis:
```python
def _slice(arr: mx.array, axis: int, start: int, length: int) -> mx.array:
    if axis == 1:
        return arr[:, start:start+length, :]
    elif axis == 2:
        return arr[:, :, start:start+length]
    raise ValueError(f"Unsupported axis: {axis}")
```

**Risk**: None. Slicing in MLX returns a view. Numerically identical.

---

**Priority 2: Replace split+stack with concatenate-slices** (Est. speedup: 10-15%, effort: 2 hours)

Before (`_apply_floors`, line 230):
```python
rows = list(mx.split(U, NVAR, axis=0))
rows[IDN] = mx.maximum(rows[IDN], RHO_FLOOR)
rows[ISR] = mx.maximum(rows[ISR], 0.0)
# ... more modifications ...
return mx.stack([r[0] for r in rows], axis=0)
```

After:
```python
rho_new = mx.maximum(U[IDN], RHO_FLOOR)
Srho_new = mx.maximum(U[ISR], 0.0)
# ... compute modifications ...
# Reconstruct only modified rows
return mx.concatenate([
    rho_new[None],       # IDN=0
    U[1:ISR],            # IMR, IMZ, IMT, IEN (indices 1-4)
    Srho_new[None],      # ISR=5
    U[ISR+1:],           # IBR, IBZ, IBT, IEE (indices 6-9)
], axis=0)
```

Each function modifies a different subset of rows, so the concatenation pattern varies. The 5 call sites:

| Function | Modifies | Untouched |
|----------|----------|-----------|
| `_apply_floors` | IDN, ISR, IEN (+ Alfven floor on IDN) | IMR, IMZ, IMT, IBR, IBZ, IBT, IEE |
| `_clamp_velocity` | IMR, IMZ, IMT, IEN | IDN, ISR, IBR, IBZ, IBT, IEE |
| `_resync_energy` | IEN | All others |
| `_geometric_sources` | Creates new array (no split+stack) | N/A |
| CT update (mlx_solver.py:716) | IBR, IBZ | All others |

**Risk**: Low. Must verify that MLX slice `U[1:5]` on a (10, nr, nz) array produces a view, not a copy. Verified: MLX slicing is lazy (like NumPy).

---

**Priority 3: Compile geometric sources and fast_magnetosonic** (Est. speedup: 3-5%, effort: 30 min)

```python
_COMPILED["geometric_sources"] = _compile_if_available(_geometric_sources_impl)
_COMPILED["fast_magnetosonic"] = _compile_if_available(_fast_magnetosonic_impl)
```

Requires extracting pure-function implementations (no `grid` object access in compiled path -- pass `inv_r` as argument).

**Risk**: Low. Verify numerical equivalence with `mx.allclose()`.

---

**Priority 4: Compile _apply_floors and _clamp_velocity** (Est. speedup: 2-3%, effort: 1 hour)

Depends on Priority 2 (eliminate split+stack first). Once refactored to pure elementwise ops, these become compile-safe.

**Risk**: Low. The Alfven-speed density floor in `_apply_floors` has a data-dependent branch (`rho_new > rho_old`) that compiles correctly as `mx.where`.

---

**Priority 5: Native MLX HLL flux** (Est. speedup: 30-50% for HLL users, effort: 4 hours)

The HLL fallback in `mlx_riemann.py:87` converts to NumPy float64, computes on CPU, converts back. Rewriting in native MLX eliminates the CPU roundtrip. However, HLLD is the default Riemann solver, so this only benefits users who explicitly select HLL or when HLLD falls back to HLL on NaN.

**Risk**: Medium. Float32 HLL requires the same numerical guards as HLLD (stable discriminant, Lax-Friedrichs fallback). The existing float64 path is numerically safe by construction.

---

#### Summary Table

| Priority | Optimization | Speedup | Effort | Risk | Changes Results? |
|----------|-------------|---------|--------|------|-----------------|
| 1 | Replace _take with slicing | 5-8% | 1h | None | No |
| 2 | Replace split+stack with concat-slices | 10-15% | 2h | Low | No |
| 3 | Compile geo sources + fast_mag | 3-5% | 30m | Low | No |
| 4 | Compile floors + velocity clamp | 2-3% | 1h | Low | No |
| 5 | Native MLX HLL | 30-50% (HLL only) | 4h | Medium | No (float32 vs float64 rounding) |

**Combined estimate**: 20-30% total speedup from priorities 1-4. Priority 5 is optional.

### 2.5 Control

#### Benchmark Methodology

**Test A: Sod shock tube (unit test, ~2 seconds)**
```python
# 1D Sod problem on (128, 1, 128) grid, 500 steps
# Measure: wall-clock time, L1(rho) error
solver = MLXMHDSolver(grid_shape=(128, 1, 128), dx=1/128, ...)
t0 = time.perf_counter()
for _ in range(500):
    solver.step(dt=1e-4)
elapsed = time.perf_counter() - t0
```

**Test B: PF-1000 discharge (production, ~200 seconds)**
```python
# Full (240, 1, 800) grid, 100,000 steps
# Measure: wall-clock time, steps/second, memory peak
```

Run each benchmark 3 times, report median. Compare before/after each optimization.

#### Verification Protocol

After each optimization:
1. Run `pytest tests/test_mlx_*.py -x -q` -- all 370+ tests must pass
2. Run Sod shock tube: L1(rho) must match pre-optimization value to < 1e-7 (float32 ulp)
3. Run 100-step PF-1000: state checksum (sum of all U components) must match pre-optimization to < 1e-4 relative
4. Profile with `mx.metal.start_capture()` / `mx.metal.stop_capture()` to verify kernel fusion

#### Regression Gate

Add to CI:
```yaml
- name: MLX perf regression
  run: |
    python3 scripts/benchmark_mlx.py --test sod --steps 500
    # Fail if >10% slower than baseline stored in data/benchmarks/mlx_sod_baseline.json
```

---

## Risk Register

| ID | Risk | Impact | Probability | Mitigation |
|----|------|--------|-------------|------------|
| R1 | MLX solver unstable at (240, 800) for >10 us (Sprint 4 electrode NaN blocker) | Calibration sweep impossible | Medium | Use HLL fallback at electrode boundary; validate solver stability before sweep |
| R2 | No experimental I(t) waveform digitized in codebase | Cannot compute NRMSE | High | Digitize Scholz (2006) Fig. 3 or use Lee-model I(t) as proxy; note that Lee-model comparison has different physics |
| R3 | Compensating errors mask physics bug | False calibration | Medium | Multi-condition validation (27 kV + 16 kV); conservation audit; physical plausibility check on optimal (fc, fm) |
| R4 | mx.compile() changes numerical results | Fail verification | Low | Run bit-exact comparison before/after each compile addition; use `mx.allclose(atol=1e-7)` |
| R5 | Slicing vs _take produces different results | Fail verification | Very Low | MLX slicing and mx.take are semantically identical for contiguous ranges |
| R6 | Memory pressure from 140 sequential MLX runs | M3 Pro 36GB OOM | Low | Call `mx.metal.clear_cache()` between runs; use reduced grid for sweep |
| R7 | Optimal (fc, fm) outside physical range | Indicates solver bug | Medium | If fc < 0.50 or fm > 0.30 is optimal, investigate numerical diffusion magnitude |

---

## References

1. **Lee S. & Saw S.H.** (2014). Pinch current and neutron yield of the PF1000. *J. Fusion Energy* 33, 319-335. -- fc, fm definitions and published PF-1000 values.

2. **Akel M. et al.** (2021). Comparison of measured and computed neutron yields from PF1000 plasma focus device operated with deuterium gas. *Radiation Physics and Chemistry* 188, 109633. -- 24-shot PF-1000 validation data, per-shot fc/fm values.

3. **Scholz M. et al.** (2006). Status of a mega-joule scale plasma focus experiment. *Nukleonika* 51(1), 79-84. -- PF-1000 circuit parameters (C, L0, R0), experimental I(t) waveform.

4. **Borges R. et al.** (2008). An improved weighted essentially non-oscillatory scheme for hyperbolic conservation laws. *J. Comput. Phys.* 227, 3191-3211. -- WENO-Z nonlinear weights.

5. **Miyoshi T. & Kusano K.** (2005). A multi-state HLL approximate Riemann solver for ideal magnetohydrodynamics. *J. Comput. Phys.* 208, 315-344. -- HLLD solver.

6. **Popovas A. et al.** (2025). DISPATCH HLLS: An entropy-stable approximate MHD Riemann solver for astrophysical flows. *arXiv:2211.02438*. -- Dual-energy entropy switching criterion.

7. **Shu C.-W.** (2009). High order weighted essentially nonoscillatory schemes for convection dominated problems. *SIAM Rev.* 51, 82-126. -- FD point-value WENO5 formulas.

8. **Damideh V.** (2025). FAETON-I plasma focus: two-step radial fitting. -- Two-step radial calibration methodology.

9. **MLX Documentation** (2026). https://ml-explore.github.io/mlx/ -- mx.compile(), Metal kernel API, array semantics.
