# DPF-Unified Forward Plan: Tier 2-4 Scaffold

**Date**: 2026-03-25
**Author**: dpf-validation-engineer (Opus)
**Methodology**: Six Sigma DMAIC planning scaffold
**Status**: Pre-implementation research — NO code changes

---

## Table of Contents

1. [Tier 2: Validation Hardening (4-6 hours)](#tier-2-validation-hardening)
2. [Tier 3: Physics Improvements (8-20 hours)](#tier-3-physics-improvements)
3. [Tier 4: Backlog (next major cycle)](#tier-4-backlog)
4. [Cross-Tier Dependencies](#cross-tier-dependencies)
5. [Risk Register](#risk-register)

---

## Tier 2: Validation Hardening

**Goal**: Wire existing validation infrastructure into CI, smoke-test all 16 presets, and establish Gribkov waveform as alternative PF-1000 reference.

**Time estimate**: 4-6 hours
**Prerequisites**: Timing error RCA complete (docs/TIMING_ERROR_RCA.md), calibration pipeline functional

### WU-2.1: Wire Validation into CI

**Scope**: Create `pytest.mark.validation` tests that run on every PR.

**Inputs**:
- `src/dpf/validation/engine_validation.py` — `run_rlc_snowplow_pf1000()` + `compare_engine_vs_experiment()`
- `src/dpf/validation/suite.py` — `ValidationSuite.validate_circuit()` with `DEVICE_REGISTRY` (4 devices: PF-1000, PF-1000-20kV, NX2, LLNL-DPF)
- `src/dpf/validation/convergence_study.py` — `run_convergence_study()` calling `app_mhd.run_mhd_simulation()`
- `app_validation.py` — `PRESET_TO_DEVICE` mapping (8 entries: pf1000, pf1000_20kv, nx2, unu_ictp, poseidon, poseidon_60kv, mjolnir, faeton)

**Deliverables**:
- [ ] `tests/test_validation_ci.py` with `@pytest.mark.validation` tests
- [ ] CI pass/warn thresholds documented
- [ ] Validation results cached for comparison across commits

**Exit Criteria**:
- `pytest -m validation` runs in < 60s on M3 Pro
- At least PF-1000 and UNU-ICTP have waveform NRMSE assertions
- Test suite reports I_peak error, t_peak error, NRMSE for all devices with waveforms

**Dependencies**: None (can start immediately)

---

### WU-2.2: Multi-Device Smoke Tests

**Scope**: Run all 16 presets through the engine, flag NaN-prone ones, establish minimum viable grid.

**Inputs**:
- `src/dpf/presets.py` — 16 presets: tutorial, pf1000, pf1000_akel, pf1000_20kv, nx2, unu_ictp, llnl_dpf, mjolnir, faeton, poseidon, poseidon_60kv, aecs_pf2, pf400j, custom, cartesian_demo, phase_p_fidelity
- `src/dpf/validation/experimental_devices.py` — 11 ExperimentalDevice objects

**Deliverables**:
- [ ] Smoke test for every preset: `run_simulation_core(preset, sim_time_us=5.0)` at 16x1x32
- [ ] Table: preset vs {NaN?, I_peak, t_peak, wall_time}
- [ ] Identified NaN-prone presets documented with root cause

**Exit Criteria**:
- All 16 presets run without NaN at some grid size
- Wall-time per preset documented
- Known-bad presets have `@pytest.mark.xfail` with reason

**Dependencies**: None

---

### WU-2.3: Gribkov Waveform as Alternative Reference

**Scope**: Wire PF-1000-Gribkov (94-point, digital oscilloscope) as an alternative validation target alongside Scholz (26-point, hand-digitized).

**Inputs**:
- `src/dpf/validation/experimental_waveforms.py` — `PF1000_GRIBKOV_T_TRIMMED`, `PF1000_GRIBKOV_I_TRIMMED` (already extracted, 90 points t>=0)
- `src/dpf/validation/experimental_devices.py` — `PF1000_GRIBKOV_DATA` (already defined, t_peak=6.39 us, I_peak=1.846 MA)
- `docs/TIMING_ERROR_RCA.md` Section 1.4 — flat-top ambiguity analysis

**Deliverables**:
- [ ] `compare_engine_vs_experiment()` extended to accept device_name="PF-1000-Gribkov"
- [ ] Validation test comparing against both Scholz AND Gribkov references
- [ ] Documentation of flat-top ambiguity (5.2-6.4 us, 1.5% variation)

**Exit Criteria**:
- NRMSE computed against Gribkov waveform
- t_peak error reframed: vs Scholz (10-14%) AND vs Gribkov (0-3%)
- Calibration pipeline (`mlx_calibration.py`) supports Gribkov as reference

**Dependencies**: WU-2.1 (test framework)

---

### Tier 2 Questions (must answer before implementation)

#### Infrastructure & Interface Questions

1. **Which validation modules are functional vs stubs?** The `__init__.py` exports from 10 modules (bennett, calibration, convergence, dynamic_zpinch, engine_validation, experimental, experimental_comparison, lee_model, magnetized_noh, pinch_physics, riemann_exact, sedov_exact, suite). Which have been exercised in actual simulations vs only unit-tested in isolation?

2. **What is the exact interface of `run_convergence_study()`?** It imports from `app_mhd.run_mhd_simulation()` — does this import work in the test environment? What's the return schema? Is `app_mhd` a Gradio app module or importable as library?

3. **What does `ValidationSuite.validate_circuit()` expect in `sim_summary`?** The code checks `"peak_current_A"`, `"peak_current_time_s"`, `"energy_conservation"`. Does `run_simulation_core()` return these keys? What about `I_pre_dip` vs `I_peak` (seen in `app_validation.py`)?

4. **`DEVICE_REGISTRY` in `suite.py` has 4 devices; `DEVICES` in `experimental.py` has more. Which is the source of truth?** The `DEVICE_REGISTRY` uses `DeviceData` (dataclass), while `experimental_devices.py` uses `ExperimentalDevice`. Are these the same? Why two registries?

5. **Where is `DEVICES` dict assembled?** `experimental.py` imports from `experimental_devices.py`. Is there a single canonical dict, or is it scattered? The `__init__.py` exports `DEVICES` from `experimental.py` — what devices are in it?

#### Waveform & Data Questions

6. **Which presets have waveform_t and waveform_I populated?** From code review: PF-1000 (measured, 26 pts), UNU-ICTP (measured, 45 pts), PF-1000-16kV (reconstructed, 25 pts), PF-1000-Gribkov (measured, 94 pts), POSEIDON-60kV (measured, 34 pts), FAETON-I (reconstructed, 25 pts), MJOLNIR (reconstructed, 25 pts). NX2, POSEIDON (40kV), and smaller devices have `waveform_provenance=""`. Is this accurate?

7. **What's the Gribkov waveform's actual t_peak?** The `PF1000_GRIBKOV_DATA` states `current_rise_time=6.39e-6` (6.39 us) and `peak_current=1.846e6` (1.846 MA). But the RCA says the flat-top spans 5.2-6.4 us with only 1.5% variation. Should t_peak be defined as the argmax (6.39 us) or the midpoint of the flat-top region (~5.8 us)?

8. **What NRMSE threshold is appropriate for CI gating?** Current best: NRMSE=0.133 vs Scholz for RLC+snowplow (engine_validation.py defaults). The MLX solver adds MHD error. Is 0.15 (warn) / 0.25 (fail) reasonable? What about per-device thresholds?

9. **Should CI gate on I_peak error, t_peak error, NRMSE, or composite J?** The calibration pipeline uses composite `J = w1*|I_err| + w2*|t_err| + w3*NRMSE`. Should CI use the same composite, or separate pass/fail per metric? What weights?

10. **What happens when a validation test fails — block merge or just warn?** Hard-gating on NRMSE blocks physics changes that temporarily degrade accuracy. Soft-gating (warn + human review) risks regression. What's the policy?

#### Performance & Platform Questions

11. **How long do validation tests take per device?** `run_rlc_snowplow_pf1000()` at dt=1e-9 for 10 us = 10,000 steps. Fast (< 1s). But MHD validation at 32x1x64 for full discharge (~10 us) takes minutes. Can validation tests fit in pre-push hook (< 30s) or only in CI?

12. **Do validation tests need MLX hardware?** The MLX solver requires Apple Silicon. CI runners may be x86. Should validation tests have a `@pytest.mark.skipif(not HAS_MLX)` guard? Should there be separate validation tracks for circuit-only vs MHD?

13. **What's the minimum grid size that doesn't NaN for each preset?** 16x1x32 may be too coarse for high-energy devices (MJOLNIR at 60 kV, POSEIDON at 60 kV). Is there a preset-specific minimum grid? Does `convergence_study.py` default to 16x1x32 (coarse)?

14. **Are there device-specific CFL or grid requirements?** MJOLNIR (2 MJ, 2.8 MA) has much higher magnetic pressure than UNU-ICTP (2.7 kJ, 169 kA). Does the CFL condition scale with device energy? Would a single grid size work for all, or do we need per-device grid presets?

#### Testing & Quality Questions

15. **What's the test isolation story?** Do validation tests modify global state (presets, solver instances)? Can they run in parallel with `pytest-xdist`? Do they require exclusive GPU access?

16. **How do we handle shot-to-shot variability?** PF-1000 Scholz vs Gribkov show different I_peak (1.87 vs 1.846 MA) and t_peak (5.8 vs 6.39 us) for the "same" device. Should validation use tolerance bands derived from shot-to-shot spread, or exact reference values?

17. **What's the regression detection sensitivity?** If a physics change shifts NRMSE from 0.133 to 0.140 (5% relative), is that signal or noise? What's the numerical noise floor for the RLC+snowplow solver?

18. **Should validation run against both Scholz AND Gribkov simultaneously?** If so, how do we reconcile different t_peak targets (5.8 us vs 6.39 us)? Report both? Use the one closer to simulation? Use the Gribkov flat-top midpoint?

19. **What `experimental_comparison.py` and `experimental_diagnostics.py` contain?** These are in the validation directory but not exported from `__init__.py`. Are they functional? Do they overlap with `experimental.py`?

20. **How does `quality_assessment.py` relate to CI validation?** It's in the validation directory. Does it compute a "quality score" that could serve as a CI metric? What does it assess?

21. **What's the `reproducibility.py` module for?** Listed in validation/ but not exported. Does it test deterministic reproducibility across runs? Could it detect non-deterministic GPU behavior?

22. **How are calibration modules (`_calibration_*.py`, 6 files, ~139 KB) used?** These are substantial. Are they invoked during validation, or only during explicit calibration runs? Would they add overhead to CI?

23. **Does `vv_report.py` generate human-readable reports?** Could it be used to auto-generate validation summaries for PRs?

24. **What is the `lee_model_comparison.py` (39 KB) test coverage?** It implements a full Lee model. Is it tested? Does it run as part of existing CI?

---

## Tier 3: Physics Improvements

**Goal**: Reduce structural timing error through density-weighted Lp, improve MHD fidelity with HLLD float64, and quantify grid contribution via convergence study.

**Time estimate**: 8-20 hours
**Prerequisites**: Tier 2 complete (validation infrastructure in place to measure improvements)

### WU-3.1: Density-Weighted Lp During Axial Rundown

**Scope**: During snowplow axial phase, use `CircuitCoupler.compute_feedback()` to get density-weighted Lp from MHD fields instead of analytical `L_coeff * z`.

**Inputs**:
- `src/dpf/engine/circuit_coupling.py` lines 60-121 — Lp handoff logic. Currently: snowplow Lp during axial, blend to MHD during radial.
- `src/dpf/circuit/coupler.py` lines 106-185 — `CircuitCoupler.compute_feedback()`. Uses density peak to find z_sheath, density-weighted r_eff, Lee formula Lp = (mu_0/2pi) * z * ln(b/r_eff). Enforces monotonicity.
- `src/dpf/fluid/snowplow.py` — `SnowplowModel.step()` returns `{"L_plasma": Lp, "dL_dt": ...}`

**Deliverables**:
- [ ] During axial rundown: optionally use `compute_feedback()` Lp instead of snowplow Lp
- [ ] Config flag: `snowplow.axial_lp_mode = "analytical" | "density_weighted"`
- [ ] Lp smoothing/monotonicity filter applied to density-weighted Lp
- [ ] Comparison: t_peak shift with analytical vs density-weighted Lp

**Exit Criteria**:
- No NaN or divergence with density-weighted Lp
- t_peak changes documented (expected: 2-5% improvement)
- Calibration impact assessed (fc/fm may need re-tuning)

**Dependencies**: WU-2.1 (need validation to measure improvement)

---

### WU-3.2: HLLD Float64 Intermediate States

**Scope**: Use float64 for HLLD star-state computation while keeping float32 for storage and flux computation.

**Inputs**:
- `src/dpf/metal/mlx_solver.py` — MLX solver currently uses HLL (HLLD hits float32 cancellation at extreme electrode B_theta)
- MEMORY.md: "HLL+PLM stable for full PF-1000 discharge on MLX. HLLD star-states still hit float32 cancellation at extreme electrode B_theta even with correct ghost cells."
- `docs/TIMING_ERROR_RCA.md` Section 5, Priority 4: "HLLD float64 intermediate states or HLL fallback near electrodes"

**Deliverables**:
- [ ] Mixed-precision HLLD: float64 for discriminant `(a^2-va^2)^2 + 4*a^2*Bt^2/rho` and star-state pressures
- [ ] Fallback to HLL where HLLD produces NaN (existing pattern from Phase O)
- [ ] Benchmark: HLLD vs HLL performance impact on MLX

**Exit Criteria**:
- HLLD runs full PF-1000 discharge without NaN on MLX
- Waveform NRMSE improves (expected: 1-3%)
- Performance regression < 20% vs HLL-only

**Dependencies**: WU-2.1 (validation), WU-3.1 is independent

---

### WU-3.3: Grid Convergence Study for t_peak

**Scope**: Run PF-1000 at 4 resolutions, compute Richardson extrapolation for t_peak and I_peak.

**Inputs**:
- `src/dpf/validation/convergence_study.py` — `run_convergence_study()` with `compute_convergence_order()`, `richardson_extrapolation()`, `grid_convergence_index()`
- Default resolutions: (16,1,32), (32,1,64), (64,1,128)

**Deliverables**:
- [ ] Convergence study at 4 resolutions: 32x1x64, 48x1x96, 64x1x128, 128x1x256
- [ ] Plot: t_peak vs N_cells (log-log)
- [ ] Richardson-extrapolated t_peak and I_peak
- [ ] GCI (Grid Convergence Index) for finest grid

**Exit Criteria**:
- Convergence order computed (expect ~1-2 for t_peak)
- GCI < 5% at finest grid (or documented why not)
- Grid contribution to timing error quantified (expected: 0.03-0.10 us)

**Dependencies**: None (can run in parallel with WU-3.1/3.2)

---

### Tier 3 Questions (must answer before implementation)

#### Density-Weighted Lp Questions

1. **Where exactly is Lp computed during snowplow axial phase?** `circuit_coupling.py` lines 75-81: `sp_result = snowplow.step(dt_sub, current)`, then `Lp_sp = sp_result["L_plasma"]`. The snowplow's `step()` returns the analytical Lee formula Lp. The MHD `compute_feedback()` is only called at line 68-71, outside the sub-step loop. Is the MHD state updated frequently enough for density-weighted Lp to be meaningful during axial rundown?

2. **Does `CircuitCoupler.compute_feedback()` work during the axial phase?** It requires `state["rho"]` to be populated. During axial rundown, is the MHD grid initialized? If the MHD solver only starts at radial phase onset, there's no density field to weight.

3. **How is the MHD state initialized during axial rundown?** If the grid is initialized with uniform fill density plus a sheath profile, does `compute_feedback()` produce a meaningful z_sheath from `argmax(col_density)`? Or does the initial uniform state confuse the density-peak detector?

4. **Would density-weighted Lp introduce noise that destabilizes the circuit ODE?** The analytical snowplow Lp is smooth (monotonically increasing). MHD-derived Lp has cell-level noise. The monotonicity filter at `coupler.py:180-184` clamps Lp >= Lp_max. Is this sufficient, or does the noisy dLp/dt still cause oscillating back-EMF?

5. **Is there a blending strategy for transitioning from analytical to density-weighted Lp?** The radial phase uses exponential blending (`circuit_coupling.py:94-102`). Should the axial phase use the same approach? What's the blend timescale?

6. **What happens to Lp at the snowplow-to-MHD handoff?** `circuit_coupling.py:85-121` shows the handoff logic: blend only during `radial` or `reflected` phase. If we add density-weighted Lp during `rundown` phase, does the handoff logic need restructuring?

7. **Does the snowplow model have a "blending" phase between rundown and radial?** `snowplow.py` has phases: "rundown", "radial", "reflected", "pinch". Is the transition from rundown to radial instantaneous (discontinuous), or is there a smooth transition?

8. **What is `self._should_use_coupler()`?** This gate determines whether `compute_feedback()` is called. Does it check for MHD grid existence, coupling mode, or something else? Would it allow density-weighted Lp during axial phase?

9. **What config flag controls the handoff mode?** `self.config.snowplow.handoff_mode` can be `"radial_mhd"`, `"full_mhd"`, or others. Is there an existing mode that enables density-weighted Lp during axial? Or do we need a new config option?

10. **If density-weighted Lp changes the sheath velocity, does it feed back to the snowplow model?** The snowplow model drives sheath position z(t). If the circuit sees a different Lp (and thus different dI/dt), the snowplow's F_mag = (mu_0/4pi)*ln(b/a)*(fc*I)^2 changes. This is a two-way coupling. Is the snowplow sub-stepping (500 steps per T_LC) sufficient to resolve this?

#### HLLD Float64 Questions

11. **What are the exact HLLD float32 cancellation sites in the MLX solver?** MEMORY.md references `metal_riemann.py:271-273` for the `dp_dt` chain rule. But the MLX solver has its own Riemann implementation. Where is the HLLD code in the MLX pipeline? Is it in `mlx_riemann.py` or similar?

12. **Can MLX do mixed precision (float64 for star-states, float32 for fluxes)?** MLX arrays have a dtype. Can we cast intermediate arrays to float64 for the HLLD discriminant computation and cast back? What's the performance cost of dtype conversions?

13. **Does MLX support float64 at all?** MLX on M3 Pro — does the Metal GPU handle float64 operations? Or does float64 force CPU fallback? This determines whether mixed precision is feasible.

14. **What's the NaN incidence rate for HLLD on PF-1000?** Is it 100% (always NaN) or intermittent? At what simulation time does it first appear? Only near electrodes, or throughout the domain?

15. **Could the existing HLL fallback pattern (Phase O) be adapted for MLX?** The Metal solver (`metal_riemann.py`) already has NaN-detected HLL fallback. Can the same pattern work in MLX? Or does MLX's functional/compiled style make per-cell branching expensive?

#### Grid Convergence Questions

16. **What grid sizes should the convergence study use?** The default in `convergence_study.py` is (16,1,32), (32,1,64), (64,1,128). Adding (128,1,256) would be ideal. But how long does 128x256 take? For full PF-1000 discharge (~10 us), is it hours?

17. **How long does each resolution take on M3 Pro with MLX?** 32x64 for full discharge: minutes? 128x256: hours? Is there a cached estimate? Does `convergence_study.py` record `wall_times`? (Yes, per the code.)

18. **What's the expected convergence order for t_peak?** HLL+PLM is formally 2nd order in space. But t_peak is an integrated quantity (cumulative error). Expected order: ~1-2. Is there a theoretical prediction for convergence of scalar diagnostics?

19. **Should the convergence study use the MLX solver or the Python engine?** The timing error was characterized with MLX. Should the convergence study use the same backend? Or should it use the Python engine (which has WENO5+HLLD) for maximum accuracy at each resolution?

20. **Does the convergence study run the full coupled simulation (circuit + snowplow + MHD)?** `run_convergence_study()` calls `run_mhd_simulation()` which goes through the full engine. This includes the snowplow model. So grid convergence measures the full system, not just MHD. Is that the right thing to study? Or should we isolate the MHD contribution?

21. **What sim_time is needed for convergence study?** Must run past t_peak (~6 us for PF-1000). The default in `convergence_study.py` is 5.0 us — that's too short! Need at least 8-10 us. Does the study auto-detect sim_time from device parameters?

22. **Can the convergence study detect oscillatory convergence?** `compute_convergence_order()` returns 0.0 if the ratio is negative (oscillatory). Is oscillatory convergence expected for t_peak due to the snowplow-MHD coupling?

23. **How do we separate grid convergence of MHD from snowplow model error?** The snowplow is 0D (no grid). Grid refinement only affects the MHD portion. But during axial phase, the snowplow dominates timing. So grid convergence may show near-zero improvement for t_peak if the snowplow is the bottleneck. How do we interpret this result?

24. **Should the convergence study also track the current dip depth and timing?** The dip (at ~7 us for PF-1000) is purely MHD-driven (radial compression). It should converge faster than t_peak. Tracking both would distinguish snowplow-limited from MHD-limited convergence.

---

## Tier 4: Backlog

**Goal**: Long-term improvements requiring significant research and/or compute investment.

**Time estimate**: 40-200+ hours total
**Prerequisites**: Tiers 2-3 complete, validated baseline established

### WU-4.1: WALRUS Fine-Tuning on DPF Training Data

**Scope**: Generate DPF training trajectories, export to Well HDF5, fine-tune 1.3B WALRUS model on M3 Pro.

**Inputs**:
- `src/dpf/ai/batch_runner.py` (382 LOC) — trajectory generation pipeline
- `src/dpf/ai/well_exporter.py` — Well HDF5 export
- `src/dpf/ai/surrogate.py` — WALRUS inference (fully implemented, 4.8 GB checkpoint)
- `src/dpf/ai/dataset_validator.py` — NaN/Inf checks, schema validation

**Deliverables**:
- [ ] 100+ DPF training trajectories in Well format
- [ ] Fine-tuned WALRUS checkpoint on DPF data
- [ ] Inference accuracy benchmark: surrogate vs full simulation

**Exit Criteria**:
- Mean L1 error < 10% on held-out DPF trajectories
- Inference time < 200ms per step on M3 Pro
- Surrogate captures I(t) waveform shape (NRMSE < 0.20 vs full sim)

**Dependencies**: Tier 2 (validation), Tier 3 (improved physics for training data quality)

---

### WU-4.2: Full 2D Axial Rundown Replacing Snowplow

**Scope**: Replace the 0D snowplow ODE during axial rundown with the full 2D MHD solver. This eliminates fc/fm dependence for the axial phase entirely.

**Inputs**:
- `src/dpf/fluid/snowplow.py` — current 0D model
- `docs/TIMING_ERROR_RCA.md` Section 5, Priority 6 — research topic, ~500-1000 LOC
- No existing DPF code has done coupled-circuit 2D axial dynamics

**Deliverables**:
- [ ] 2D axial grid initialization with electrode geometry
- [ ] Electrode boundary conditions for axial sheath propagation
- [ ] Circuit coupling via MHD-derived Lp throughout axial phase
- [ ] Comparison: 0D snowplow vs 2D axial dynamics

**Exit Criteria**:
- 2D axial simulation runs without NaN
- I(t) waveform qualitatively matches experimental shape
- fc/fm no longer needed (simulation determines mass sweeping naturally)

**Dependencies**: WU-3.1 (density-weighted Lp as stepping stone), WU-3.3 (grid convergence establishes baseline)

---

### WU-4.3: AMR Adaptive Mesh Refinement

**Scope**: Implement block-based AMR for concentrating resolution at the sheath and pinch column.

**Inputs**:
- Currently: uniform grid only. No AMR infrastructure exists.
- CLAUDE.md: "AMR (`src/dpf/experimental/amr/`) — NOT IMPLEMENTED. Design required (~60-80h)."
- Design space: patch-based (Berger-Oliger) vs block-based (Athena++ style)

**Deliverables**:
- [ ] AMR design document (algorithm, data structures, refinement criteria)
- [ ] Block-based AMR implementation with 2 refinement levels
- [ ] Refinement criteria: density gradient and magnetic pressure gradient
- [ ] Conservation enforcement at fine-coarse boundaries

**Exit Criteria**:
- 2x refinement at sheath achieves same accuracy as 2x global grid at half the cost
- Conservation errors < 1e-6 at fine-coarse interfaces
- Sheath resolution maintained throughout rundown

**Dependencies**: WU-3.3 (grid convergence quantifies uniform grid limitations)

---

### WU-4.4: Memory System Repair

**Scope**: Fix the `memory-pruner` tool and resolve accumulated state inconsistencies.

**Inputs**:
- `~/bin/memory-pruner` exists on PATH
- MEMORY.md: caps at 200 lines (MEMORY.md), 100 entries (observations.md), 50 entries (logbook.md)
- Memory files may have exceeded caps

**Deliverables**:
- [ ] `memory-pruner` bug diagnosed and fixed
- [ ] Stale observations pruned
- [ ] Topic files consolidated
- [ ] Archive pipeline functional

**Exit Criteria**:
- `memory-pruner --dry-run` runs cleanly
- All memory files within caps
- Archives generated for overflow

**Dependencies**: None (independent of Tiers 2-3)

---

### Tier 4 Questions (must answer before implementation)

#### WALRUS Fine-Tuning Questions

1. **How many DPF training trajectories do we need?** WALRUS was pretrained on 19 scenarios with thousands of trajectories each. For fine-tuning on a single domain (DPF), how many trajectories give diminishing returns? Literature suggests 50-200 for fine-tuning, but DPF has narrow physics (all cylindrical, all Z-pinch).

2. **What's the `batch_runner.py` actual status?** It's 382 LOC and listed as "Ready" in MEMORY.md. But has it been tested end-to-end? Does `run_batch()` actually produce valid Well HDF5 files? What parameter ranges does it sweep?

3. **Is the 4.8 GB WALRUS checkpoint compatible with current torch version?** WALRUS pins `torch==2.5.1`. If the DPF venv has a different torch, does loading fail? Is the separate venv documented and tested?

4. **What grid size does WALRUS require?** Minimum 16x16x16 3D (from CLAUDE.md). DPF is 2D cylindrical (nr x 1 x nz). How does a 2D cylindrical grid map to WALRUS's 3D Cartesian expectation? Does `field_mapping.py` handle this?

5. **How does cylindrical DPF data map to WALRUS's Cartesian 3D?** DPF has (r, theta, z) with ny=1 (axisymmetric). WALRUS expects (x, y, z). Options: (a) pad ny to 16 with copies, (b) treat r as x and z as y with nz=1, (c) generate synthetic 3D data by rotating 2D solution. Which approach loses least physics?

6. **What fields should the training data include?** WALRUS handles scalar (t0), vector (t1), and tensor (t2) fields. DPF has: rho, pressure, Te, Ti (scalars); velocity, B (vectors); plus circuit current (non-spatial scalar). Does the Well schema handle non-spatial time series?

7. **What's the compute budget for trajectory generation?** 100 trajectories at 32x64 for 10 us each: ~100 * 5 min = ~8 hours on M3 Pro with serial execution. Parallelism with batch_runner? Memory constraints with parallel runs?

8. **What boundary conditions should be declared in Well format?** DPF has: wall at r=0 (axis), conducting wall at r=b (cathode), electrode at z=0 (anode), open at z=L. Well encodes: WALL=0, OPEN=1, PERIODIC=2. No "conducting wall" or "electrode" BC type in Well.

9. **Should fine-tuning use delta prediction or absolute?** WALRUS is designed for delta prediction (Δu). DPF dynamics have a strong directional trend (current rise → peak → dip). Does delta prediction handle this well, or would absolute prediction be better for circuit-coupled systems?

10. **What training hyperparameters for Apple Silicon?** Batch size 1, gradient accumulation 4-8, gradient checkpointing, AMP disabled (MPS incompatible). LoRA or full fine-tuning? Memory budget: 36 GB total, ~19-25 GB for LoRA. Is that feasible during active development (Chrome, VS Code, etc. consuming memory)?

#### Full 2D Axial Rundown Questions

11. **What would "full 2D axial rundown" look like architecturally?** Currently: snowplow model drives z(t) analytically, MHD grid handles radial compression. The 2D axial approach would need the MHD grid to start at z=0 and propagate the sheath axially. This means the grid must cover the full axial domain from the start. Memory: 128x1x512 at float32 = ~10 MB per variable, ~100 MB total. Feasible?

12. **Would it replace SnowplowModel entirely or run alongside it?** Options: (a) full replacement — MHD drives everything from t=0, (b) hybrid — snowplow for initial few us, transition to MHD, (c) validation mode — run both and compare. Which minimizes risk?

13. **What boundary conditions are needed for axial sheath propagation in 2D MHD?** The sheath is driven by J x B force from the radial current flowing between electrodes. This requires: conducting electrode BCs (B_theta = mu_0*I/(2*pi*r) at both axial boundaries), a fill-gas initial condition ahead of the sheath, and a swept region behind. How do we handle the "break" where the sheath lifts off the insulator?

14. **Has any DPF code ever done coupled-circuit 2D axial dynamics?** Literature search needed. Most DPF codes use 0D snowplow (Lee model) or start MHD at the radial phase. Some FLASH/PLUTO simulations do full 2D/3D but with prescribed current drive (not self-consistent circuit coupling). This would be a novel contribution if successful.

15. **What's the expected performance impact?** MHD for the full axial phase (~5 us) instead of a few ODE steps. At 128x512 with CFL-limited dt, this could be thousands of MHD steps. Estimate: 10-100x slower than snowplow for the axial phase. Total sim time increase: 2-10x.

#### AMR Questions

16. **What's the AMR design space — patch-based or block-based?** Berger-Oliger (patch-based): flexible, complex bookkeeping. Athena++ (block-based): simpler, aligned with existing grid structure. FLASH (paramesh, block-based): well-tested for MHD. MLX arrays are fixed-size — does block-based AMR work with MLX?

17. **Does MLX support non-uniform grids or variable-size arrays?** MLX is designed for fixed-shape tensor operations. AMR requires different-sized arrays per refinement level. Options: (a) fixed max-level allocation with masking, (b) separate MLX arrays per level, (c) CPU-only AMR with MLX for per-block compute.

18. **What refinement criteria make sense for DPF?** Density gradient (sheath location), magnetic pressure gradient (pinch compression), or both? Vorticity? Temperature gradient? How many cells per sheath width is sufficient (3-5)?

19. **What's the conservation enforcement strategy at fine-coarse boundaries?** Flux-matching at level interfaces is critical for MHD conservation. Athena++ uses "flux correction" where fine-level fluxes replace coarse-level fluxes at interfaces. Can we adapt this for our MLX solver?

20. **What's the expected speedup from AMR vs uniform grid?** The sheath occupies ~5% of the domain volume. 2-level AMR with 2x refinement at the sheath would give ~4x fewer cells than a uniformly-fine grid. But AMR overhead (tagging, communication, level bookkeeping) reduces the net gain. Estimate: 2-3x effective speedup.

#### Memory System Questions

21. **What's the memory-pruner bug?** The script exists at `~/bin/memory-pruner`. What error does it produce? Is it a Python script, shell script, or binary? What dependencies does it have?

22. **How much has MEMORY.md grown beyond the 200-line cap?** Current state unknown. Need to count lines and identify what can be consolidated into topic files.

23. **Are observations.md entries dated and categorized?** The format should be `### [YYYY-MM-DD] -- [project]\n[one-line]`. Are all entries in this format? How many are DPF-specific vs cross-project?

24. **What archives exist and are they being consumed?** `memory/archive/` should contain monthly logbook archives. Are they being generated? Is the 6-month retention policy enforced?

25. **Does the pruner handle deduplication?** If the same observation was written in multiple sessions, does the pruner detect and merge duplicates?

---

## Cross-Tier Dependencies

```
Tier 2 (Validation)
  WU-2.1 (CI tests) ──────────┐
  WU-2.2 (smoke tests) ───────┼──> Tier 3 baseline
  WU-2.3 (Gribkov waveform) ──┘
                                │
Tier 3 (Physics)                │
  WU-3.1 (density-weighted Lp) ←┘
  WU-3.2 (HLLD float64) ──────────> Independent
  WU-3.3 (grid convergence) ──────> Independent
                                │
Tier 4 (Backlog)                │
  WU-4.1 (WALRUS) ←────────────┤─── needs Tier 2+3 for quality training data
  WU-4.2 (2D axial) ←──────────┤─── needs WU-3.1 as stepping stone
  WU-4.3 (AMR) ←───────────────┘─── needs WU-3.3 to quantify grid limits
  WU-4.4 (memory repair) ─────────> Independent (can do anytime)
```

### Critical Path

```
WU-2.1 → WU-2.3 → WU-3.1 → WU-4.2
         WU-2.2 → WU-3.3 → WU-4.3
                   WU-3.2 (parallel, independent)
         WU-4.4 (anytime)
```

---

## Risk Register

| ID | Risk | Probability | Impact | Mitigation |
|----|------|-------------|--------|------------|
| R1 | Density-weighted Lp destabilizes circuit ODE during axial phase | Medium | High | Use analytical Lp as fallback; exponential blending with long tau; monotonicity filter |
| R2 | HLLD float64 on MLX forces CPU fallback, killing performance | Medium | Medium | Benchmark early; if MLX float64 is CPU-only, use HLL+PLM as default and HLLD only for production V&V |
| R3 | Grid convergence study shows t_peak is snowplow-limited (no grid dependence) | High | Low | This is actually useful information — confirms WU-3.1 and WU-4.2 are the right priorities |
| R4 | WALRUS fine-tuning OOMs on M3 Pro 36 GB | Medium | High | Use LoRA (rank 16-32) instead of full fine-tuning; gradient checkpointing; batch size 1 |
| R5 | 2D axial rundown requires novel research with uncertain timeline | High | Medium | Treat as research spike first (2-4 hours); document findings; proceed only if viable |
| R6 | Validation CI tests too slow for pre-push hook | Medium | Low | Split into fast (circuit-only, < 5s) and slow (MHD, < 300s) tiers |
| R7 | Shot-to-shot variability (Scholz vs Gribkov) makes validation thresholds ambiguous | Medium | Medium | Use both references; report wider tolerance; document flat-top ambiguity |
| R8 | Multi-device smoke tests reveal widespread NaN issues at 16x1x32 | Low | Medium | Use per-device minimum grids; document known-bad configurations |

---

## Appendix: Device-Waveform Availability Matrix

| Preset | Device Name | Waveform Available | Points | Provenance | I_peak | t_peak |
|--------|-------------|--------------------|--------|------------|--------|--------|
| pf1000 | PF-1000 | Yes | 26 | measured (hand-digitized) | 1.87 MA | 5.8 us |
| pf1000 | PF-1000-Gribkov | Yes | 90 (trimmed) | measured (digital osc.) | 1.846 MA | 6.39 us |
| pf1000_20kv | PF-1000-16kV | Yes | 25 | reconstructed | 1.2 MA | ~6.0 us |
| unu_ictp | UNU-ICTP | Yes | 45 | measured (digital osc.) | 169 kA | 2.2 us |
| poseidon_60kv | POSEIDON-60kV | Yes | 34 | measured (IPFS) | 3.19 MA | 1.98 us |
| faeton | FAETON-I | Yes | 25 | reconstructed | 998 kA | ~3.6 us |
| mjolnir | MJOLNIR | Yes | 25 | reconstructed | 2.8 MA | ~5.0 us |
| nx2 | NX2 | No | — | — | 400 kA | 1.8 us |
| poseidon | POSEIDON | No | — | — | 2.6 MA | 5.0 us |
| pf1000_akel | (Akel 24-shot) | Partial | — | statistical | 1.2 MA | varies |
| llnl_dpf | LLNL-DPF | No | — | — | — | — |
| aecs_pf2 | AECS-PF2 | No | — | — | — | — |
| pf400j | PF-400J | No | — | — | — | — |
| tutorial | (synthetic) | N/A | — | — | — | — |
| cartesian_demo | (demo) | N/A | — | — | — | — |
| phase_p_fidelity | (benchmark) | N/A | — | — | — | — |

**Measured waveforms usable for NRMSE validation**: PF-1000 (Scholz), PF-1000 (Gribkov), UNU-ICTP, POSEIDON-60kV
**Reconstructed waveforms (use with caution)**: PF-1000-16kV, FAETON-I, MJOLNIR
**No waveform (I_peak/t_peak only)**: NX2, POSEIDON, others
