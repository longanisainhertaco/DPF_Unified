# Scaffold Review -- Six Sigma DMAIC + FMEA Analysis

**Reviewer**: DPF Validation Engineer (Opus)
**Date**: 2026-03-26
**Documents reviewed**: AMR_DESIGN_SCAFFOLD.md, PIC_VALIDATION_SCAFFOLD.md, HALL_MHD_MLX_DESIGN.md, THOMSON_SCATTERING_DESIGN.md

---

## 1. Executive Summary

Four research scaffolds were reviewed for implementation readiness. Hall MHD (score 8/10) is nearly ready -- a confirmed unit-conversion bug, clear fix, and small scope make it the highest-confidence item. Thomson Scattering (7/10) is well-scoped but depends on an Abel transform that has never been tested at the spectral level. AMR (6/10) is architecturally sound but LOC estimates are 40-60% too low based on comparable implementations, and cylindrical refluxing has no reference implementation to copy from. PIC Validation (5/10) has the most thorough inventory of any scaffold but exposes 8 known bugs in production code, 3 of which block end-to-end testing. The scaffolds have no circular dependencies but share a common gap: none addresses how existing 4,900+ tests remain green during implementation. Recommended order: Hall MHD -> Thomson -> PIC V1-V2 -> AMR Phase A -> PIC V3-V4 -> AMR Phase B-D.

---

## 2. AMR Design Scaffold

### 2.1 DMAIC

**Define**: Problem statement is clear -- add adaptive mesh refinement to the MLX cylindrical MHD solver. Success criteria are implicit (AMR runs without breaking uniform-grid mode) but not numerically specified. Missing: what effective speedup or resolution gain constitutes "success"? A 2-level AMR that is slower than a uniform fine grid is a failure, but no wall-clock target is stated.

**Measure**: LOC estimates (1,150-1,550 total) are optimistic.
- Phase A "400-500 LOC" for block data structures + ghost exchange + prolongation + restriction + solver integration. The existing `static_refinement.py` is 562 LOC and only does single-level re-simulation with no ghost exchange. A block-structured AMR with ghost exchange, prolongation, restriction, and solver wiring is closer to 700-900 LOC based on MPI-AMRVAC's Python prototype and Parthenon's block management.
- Phase C "300-400 LOC" for cylindrical refluxing. Standard Cartesian refluxing in AMReX is ~600 LOC (C++). Cylindrical r-weighted refluxing does not exist in any open-source Python implementation. This is research-grade code, not port-from-reference. Realistic estimate: 500-700 LOC.
- Test LOC "300-400 per phase" is reasonable.
- **Revised total estimate**: 1,800-2,600 LOC (production) + 1,200-1,600 LOC (tests). Calendar: 10-16 sessions, not 6-10.

**Analyze** (top 3 risks, Pareto):
1. **Cylindrical refluxing correctness (60% of failure probability)**: No reference implementation exists for r-weighted refluxing in Python. The volume elements `2*pi*r*dr*dz` create non-trivial geometry factors at every coarse-fine face. Getting this wrong silently violates conservation -- the error scales with sheath velocity and is largest exactly where accuracy matters most. This is the Pareto dominant risk.
2. **`mx.compile()` shape constraints (20%)**: The pre-allocated MAX_BLOCKS=16 with masking approach is untested. If the compiled kernel's performance degrades significantly with inactive blocks (wasted FLOPS), the AMR speedup evaporates. No profiling data supports the claim that masking is efficient.
3. **Ghost exchange at physical boundaries (15%)**: The scaffold describes 3 ghost-fill sources (neighbor copy, prolongation, physical BC) but does not specify the priority order when a ghost cell could be filled by multiple sources. Electrode BCs at the axis (r=0) interact with prolongation from a coarser level in ways that are geometry-specific and not addressed.

**Improve**:
- Risk 1 mitigation: Write a standalone cylindrical refluxing test (known analytical solution: cylindrical shock reflecting off axis) BEFORE integrating into AMR. This isolates the geometry factors from the block management complexity. ~80 LOC test.
- Risk 2 mitigation: Profile a simple benchmark -- compiled MLX kernel with 16 pre-allocated slots, 4 active, vs. 4-slot kernel. If masking overhead > 30%, reduce MAX_BLOCKS or use separate compiled functions per active count.
- Risk 3 mitigation: Define explicit priority order in the design: physical BC > same-level neighbor > prolongation. Document it.

**Control**:
- Conservation audit at every phase: `|delta_mass/mass| < 1e-8` per step for Phase A-B (no refluxing), `< 1e-12` for Phase C+.
- Regression gate: full test suite must pass with `use_amr=False` (default) after every phase.
- Wall-clock benchmark: AMR at 2 levels must be faster than uniform fine grid for the standard PF-1000 test case, or the feature is not promoted from experimental.

### 2.2 FMEA

| # | Failure Mode | Severity | Occurrence | Detection | RPN | Mitigation |
|---|-------------|----------|------------|-----------|-----|------------|
| 1 | Cylindrical refluxing wrong r-weighting | 9 | 7 | 4 (mass drift detectable) | 252 | Standalone reflux test with analytical solution |
| 2 | mx.compile recompiles on regrid (shape change) | 7 | 5 | 3 (obvious perf drop) | 105 | Pre-allocate fixed shape, verify no recompile via mx.compile stats |
| 3 | Ghost exchange order at axis (r=0) produces negative density | 8 | 4 | 5 (NaN propagates) | 160 | Priority order spec + axis-specific test case |
| 4 | Prolongation operator not div(B)-consistent | 7 | 6 | 6 (div(B) grows slowly) | 252 | Use Dedner GLM at boundaries (stated); monitor max(div B) per step |
| 5 | Regrid triggers Python GC stall mid-simulation | 5 | 3 | 7 (intermittent slowdown) | 105 | Profile regrid cost; pre-allocate all blocks at init |

### 2.3 Missing Research

- No reference implementation for cylindrical r-weighted refluxing in any GPU code. MPI-AMRVAC does Cartesian + spherical, not cylindrical. Athena++ does cylindrical AMR but in C++ with CT (not Dedner). The scaffold claims MPI-AMRVAC validates the Dedner-at-boundaries approach, but MPI-AMRVAC's cylindrical support with AMR is untested in their own test suite (Keppens 2023, Table 2 lists only Cartesian and spherical AMR tests).
- Block size 32x64: no profiling data. This choice "matches typical nr/nz ratio" but the optimal block size for MLX Metal kernels depends on warp size (32 threads on Apple GPU) and L1 cache (48KB per CU on M3). 32x64 = 2,048 cells per block x 10 NVAR x 4 bytes = 80KB, which exceeds L1. Consider 16x32 (20KB, fits L1) as an alternative.
- Parthenon-VIBE arXiv:2509.19701 -- this arXiv ID is from 2025 and may not exist yet (the 2509 prefix implies September 2025). If this is a fabricated reference, the Amdahl bottleneck analysis loses its empirical grounding. Verify this paper exists.

---

## 3. PIC Validation Scaffold

### 3.1 DMAIC

**Define**: Exceptionally well-defined. The inventory of tested vs. untested functions is the most thorough of any scaffold. The 28+6+6+3 test progression is clear. Success criteria are quantitative (Landau damping rate within 10%, gyration energy conserved to 1e-8, Yn within order of magnitude).

**Measure**:
- LOC estimates are credible. 1,400 LOC for 43 tests = ~33 LOC/test, consistent with existing test files in the codebase (`test_pic_validation.py` has 12 tests in ~400 LOC = 33 LOC/test).
- Calendar estimate "7-11 days" is aggressive given the 8 known bugs, 3 of which (sub-cycling, E-field approximation, self-collision bias) require production code changes that need re-validation. Realistic: 12-18 days if bug fixes are included.
- Bug fix LOC estimates are reasonable (total ~160 LOC for 6 fixes).

**Analyze** (top 3 risks, Pareto):
1. **PIC sub-cycling not implemented (50% of failure probability)**: Without sub-cycling, Boris push under-resolves gyration at DPF B-field strengths (10T). V3 and V4 tests will produce physically meaningless results. This is correctly identified as "most likely blocker." The ~40 LOC estimate is plausible but the fix touches `push_particles()`, which is the untested integration method. Testing the fix requires V1 tests to already be written. Circular dependency.
2. **E-field missing Hall term (25%)**: `E = -v x B` omits Hall and pressure gradient terms. At pinch, Hall E can equal convective E. This means ALL PIC trajectories at pinch conditions are wrong by up to 100%. The scaffold defers this to "Phase V3 should measure E_Hall magnitude" but offers no fallback if it is large. Adding the Hall term requires the Hall MHD scaffold to be implemented first -- a cross-scaffold dependency not explicitly stated.
3. **Reflecting BC inflates Yn (15%)**: Absorbing BC fix is listed as "MEDIUM" priority but directly affects the V4 end-to-end Yn result. A factor-of-2-5x overestimate in Yn undermines the entire validation against Gribkov (2007). This should be HIGH priority, completed before V4.

**Improve**:
- Risk 1: Implement sub-cycling (40 LOC) as the FIRST bug fix, before V1 unit tests. Use the existing Nanbu scatter sub-cycling pattern from MEMORY.md ("Sub-cycle expensive operator-split steps: N = ceil(dt_mhd/dt_res), capped at 20").
- Risk 2: Add Hall term dependency to the critical path diagram. If Hall MHD (Phase H1) is not complete, V3 must use a reduced-accuracy E-field with a documented known limitation.
- Risk 3: Promote absorbing BC to HIGH priority. Implement before V4.

**Control**:
- Energy conservation tracked per test phase: V1 (kernel-level, exact), V2 (integration, < 1e-8 per gyroperiod), V3 (MHD-coupled, < 5%), V4 (full, < 10%).
- Yn validation range: 10^10 - 10^12 for PF-1000 27kV. Below 10^9 or above 10^13 is a failure.
- Regression: existing 24 PIC tests (7 + 12 + 5) must remain green throughout.

### 3.2 FMEA

| # | Failure Mode | Severity | Occurrence | Detection | RPN | Mitigation |
|---|-------------|----------|------------|-----------|-----|------------|
| 1 | Boris push under-resolved (no sub-cycling) | 9 | 9 | 2 (gyration visibly wrong) | 162 | Implement sub-cycling first; 1000-gyroperiod conservation test |
| 2 | E-field missing Hall term corrupts trajectories at pinch | 8 | 7 | 6 (subtle trajectory error) | 336 | Measure E_Hall/E_conv ratio in V3; add Hall if > 10% |
| 3 | Reflecting BC overestimates confinement | 7 | 9 | 5 (Yn too high but plausible) | 315 | Implement absorbing BC before V4; compare Yn with/without |
| 4 | Nanbu self-collision bias (same-array mutation) | 6 | 9 | 7 (thermalization rate subtly wrong) | 378 | Copy velocity array before Nanbu call; 10 LOC |
| 5 | Esirkepov dt mismatch breaks charge conservation | 8 | 5 | 3 (continuity test catches it) | 120 | 5 LOC fix in V1; continuity equation test validates |

### 3.3 Missing Research

- The Landau damping test (Section 3.1) requires a 1D Poisson solver not in the codebase. The scaffold estimates ~80 LOC for this solver. This is plausible for a direct tridiagonal solve, but periodic boundary conditions with charge neutralization add complexity. Consider using FFT-based Poisson (`np.fft.fft` -> divide by k^2 -> `np.fft.ifft`) which is ~15 LOC and trivially correct.
- Pasternak et al. (2024) is cited for beam energy spectra comparison but no specific figure or data points are given. Need to extract the specific energy distribution for comparison.
- The "10,000 particles" for V4 may be insufficient for converged Yn. Schmidt (2012) used ~10^6 macro-particles. At 10K, the statistical noise on Yn is ~1/sqrt(N_reactions) which for beam-target with small cross-sections could be enormous. The scaffold should specify a convergence study: run with 10K, 50K, 100K and verify Yn converges.

---

## 4. Hall MHD MLX Design

### 4.1 DMAIC

**Define**: Excellent. The problem is precisely stated: a confirmed unit-conversion bug (missing mu_0 factor) in existing code, plus missing whistler CFL and sub-cycling. The fix is mathematically derived in the document. Success criteria are implicit (cross-backend parity with Metal implementation, whistler wave dispersion test).

**Measure**:
- Total ~235 LOC is credible. The existing `apply_hall_mhd` is 65 LOC; modifications are surgical.
- Calendar estimate is not explicitly stated but implied as ~1 day across 5 phases. This is realistic given the small scope.
- The HL unit derivation in Section 2 is correct. The document works through the algebra and arrives at the right answer: `E_H = (J_HL x B_HL) * mu_0 / (n_e * e)`. The factor of mu_0 is indeed missing in the current code.

**Analyze** (top 3 risks, Pareto):
1. **Whistler CFL kills performance in early rundown (50%)**: The document correctly identifies that dt_hall << dt_mhd by 1000x in pre-pinch conditions (n_e ~ 10^20, B ~ 1T). Capping sub-cycles at 20 means the Hall update is only accurate to 20/1000 = 2% of the required temporal resolution. This is the dominant risk. The cap-at-20 pattern from resistive diffusion is cited, but resistive diffusion's CFL mismatch is typically 5-10x, not 1000x.
2. **Existing Hall tests pass for wrong reasons (30%)**: The document notes "existing tests may not catch because they test 'Hall modifies B' not 'by how much'." This means the unit bug has been hiding in production, and existing tests provide false confidence. After the fix, E_Hall changes by a factor of ~1.26e-6 (mu_0). Any test that previously "passed" was accepting a result that was 10^6 wrong. This suggests the Hall module has NEVER been meaningfully active in any simulation.
3. **`ne_min` vs `ne_max` in CFL formula (15%)**: Line 282 of the proposed `hall_whistler_dt` uses `ne_min = float(mx.max(ne).item())` with the comment "use max(ne) for most restrictive." But this is labeled `ne_min`. The whistler CFL `dt ~ dx^2 * ne * e * mu_0 / B` is MOST restrictive where ne is SMALLEST and B is LARGEST. The code should use `mx.min(ne)` (excluding vacuum), not `mx.max(ne)`. This is a bug in the scaffold design itself.

**Improve**:
- Risk 1: Increase max_subcycles to 100 for Hall (not 20). At 1000x mismatch, 100 sub-cycles gives 10% accuracy. Alternatively, use implicit sub-cycling (backward Euler for the Hall term) which is unconditionally stable. The implicit approach adds ~50 LOC (solve tridiagonal system) but eliminates the sub-cycling bottleneck entirely. This should be a Phase H2b option.
- Risk 2: Write the whistler wave dispersion test (H5) BEFORE fixing the bug. Run it on the current code to confirm it fails (expected: dispersion relation off by mu_0). Then fix and re-run. This is the "red-green" TDD pattern.
- Risk 3: Fix the `ne_min` bug in the scaffold. Use `ne_min = float(mx.min(mx.where(rho > 1e-10, ne, 1e30)).item())` to exclude vacuum cells.

**Control**:
- Cross-backend parity: `L1(dB_mlx - dB_metal) < 1e-3` relative.
- Whistler dispersion: measured phase speed within 10% of analytical omega/k.
- Energy conservation: Hall term conserves total energy (magnetic + kinetic) to < 1e-6 relative over 100 steps.

### 4.2 FMEA

| # | Failure Mode | Severity | Occurrence | Detection | RPN | Mitigation |
|---|-------------|----------|------------|-----------|-----|------------|
| 1 | Whistler CFL 1000x mismatch causes inaccurate sub-cycling | 7 | 8 | 4 (dispersion test catches) | 224 | Increase cap to 100 or add implicit option |
| 2 | ne_min/ne_max CFL formula bug (scaffold itself wrong) | 8 | 10 | 3 (test will show wrong CFL) | 240 | Fix formula before implementation |
| 3 | Hall inactive in production (mu_0 error = 10^6 wrong) | 6 | 10 | 8 (only detectable by quantitative test) | 480 | Write quantitative dispersion test, not just "modifies B" |
| 4 | mx.roll boundary wrap injects unphysical Hall E at domain edges | 6 | 5 | 5 (boundary artifacts) | 150 | Zero-pad or one-sided stencil at physical boundaries |
| 5 | NaN clamp in H3 masks real bugs | 5 | 4 | 8 (NaN masked, underlying issue persists) | 160 | Log clamped cell count; fail if > 1% of cells clamped |

### 4.3 Missing Research

- The document claims "Float32 is safe for Hall MHD at DPF conditions." The analysis is correct for the magnitudes but ignores the sub-cycling accumulation. With 100 sub-cycles per MHD step and 10^5 MHD steps, that is 10^7 Hall updates. Each accumulates eps_32 * |dB| ~ 1.2e-7 * 4e-3 = 4.8e-10 per step. Over 10^7 steps: ~5e-3 accumulated error on B. For B ~ 50T (HL units ~4.5e4), this is 1e-7 relative -- acceptable, but should be stated.
- The Nernst advection (out of scope) is mentioned but not analyzed for whether it should be included. In DPF pinch conditions with steep Te gradients, Nernst advection of B can be comparable to Hall advection. The document should note this as a limitation for physics fidelity.

---

## 5. Thomson Scattering Design

### 5.1 DMAIC

**Define**: Clean problem statement. Three functions (`thomson_spectrum`, `thomson_line_integrated`, `fit_te_ne_v`) with clear I/O specs. Success criteria: spectral width within 1% of analytical, Doppler shift correct, Abel roundtrip consistent, fit roundtrip within 5%.

**Measure**:
- 200 LOC for core module is plausible. The `interferometry.py` it depends on is 161 LOC for a simpler diagnostic.
- 80 LOC for tests is LOW. Five analytical tests at ~80 LOC total = 16 LOC/test. The existing `test_pic_validation.py` averages 33 LOC/test. More realistic: 120-150 LOC.
- The Salpeter correction (alpha > 0.5) at "~50 LOC" is the most uncertain estimate. The full Salpeter form factor requires the plasma dispersion function (Fried-Conte Z-function), which is a complex-valued integral. `scipy.special` does not include it. Either approximate (Pade) or implement (~80-100 LOC for the Z-function alone).

**Analyze** (top 3 risks, Pareto):
1. **Fried-Conte Z-function not available (40%)**: The collective regime (alpha > 1) requires the plasma dispersion function Z(zeta) = (1/sqrt(pi)) * integral(exp(-t^2)/(t-zeta) dt). This is NOT in scipy. The scaffold mentions "Gaussian approximation with Sheffield correction factor" but this is only valid for alpha ~ 0.5-2, not alpha >> 1. For alpha > 2 (collective regime), the full Z-function is needed. Options: (a) implement Pade approximation (~30 LOC, Hui/Armstrong 1978), (b) use `scipy.special.wofz` (Faddeeva function, which is related by Z(zeta) = i*sqrt(pi)*w(zeta)), (c) limit to alpha < 2 only. Option (b) is the right answer and is ~5 LOC, but the scaffold does not mention it.
2. **Abel transform not tested for spectral use (30%)**: The existing `abel_transform` in `interferometry.py` integrates ne(r). Using it for spectral emissivity requires calling it M times (once per wavelength bin). With M=100 wavelength bins and nr=64, this is 100 separate Abel transforms. The function uses O(nr^2) quadrature (matrix K), so total cost is O(M * nr^2) = 100 * 4096 = 400K operations -- fast. But the Abel transform has never been tested with profiles other than density. If the emissivity has sharper features (e.g., at alpha ~ 1 transition), the quadrature may need more radial points.
3. **Te input ambiguity (20%)**: The design says "Te (nr, nz): electron temperature [K]" but the MLX solver's two-temperature module stores Te in eV internally, and the engine state dict uses various conventions. The scaffold should specify exactly which state dict key provides Te and in what units, for each backend.

**Improve**:
- Risk 1: Use `scipy.special.wofz` (Faddeeva function). `Z(zeta) = i * sqrt(pi) * wofz(zeta)`. This is exact, vectorized, and adds 0 external dependencies. Update the scaffold to reference this.
- Risk 2: Add a test that applies Abel transform to a known emissivity profile (e.g., Gaussian in r) and compares against analytical line integral.
- Risk 3: Document the Te extraction chain: `state["Te"]` in K -> `Te_eV = state["Te"] * k_B / eV`. If `state["Te"]` is not present, derive from pressure: `Te_eV = pressure / (2 * ne * eV)`.

**Control**:
- Spectral width test: `|delta_th_measured / delta_th_analytical - 1| < 0.01`.
- Roundtrip test: `|Te_fit / Te_input - 1| < 0.05`, `|ne_fit / ne_input - 1| < 0.05`.
- No NaN/Inf in output for any physical input (ne > 0, Te > 0).

### 5.2 FMEA

| # | Failure Mode | Severity | Occurrence | Detection | RPN | Mitigation |
|---|-------------|----------|------------|-----------|-----|------------|
| 1 | Missing plasma dispersion function for collective regime | 7 | 6 | 2 (import error or wrong spectrum) | 84 | Use scipy.special.wofz (Faddeeva) |
| 2 | Abel transform inaccurate for sharp spectral emissivity | 5 | 4 | 5 (subtle broadening of line-integrated spectrum) | 100 | Test with known analytical profile |
| 3 | Te unit mismatch (K vs eV vs state dict convention) | 8 | 5 | 3 (obviously wrong spectral width) | 120 | Document Te extraction chain; unit test with known Te |
| 4 | Regime discontinuity at alpha = 0.5 and alpha = 2 | 4 | 5 | 4 (jump visible in sweep test) | 80 | Use smooth blending function between regimes |
| 5 | scipy.optimize.curve_fit diverges on noisy synthetic spectrum | 5 | 4 | 3 (fit returns NaN or absurd values) | 60 | Provide initial guesses from moments; bound parameters |

### 5.3 Missing Research

- `scipy.special.wofz` should be the default for the plasma dispersion function. This eliminates the entire "Salpeter correction approximation" path and gives exact results at all alpha.
- Decker et al. (1996) is the only cited DPF Thomson measurement. The scaffold should note whether the NX-1 device parameters are in the codebase presets. If not, there is no device-specific validation target.
- The scattering angle default of 90 degrees is standard but should be configurable per-chord for future diagnostic design optimization.

---

## 6. Cross-Scaffold Dependency Analysis

### 6.1 Dependency Graph

```
Hall MHD (H1-H5)
    |
    v
PIC Validation (V3: E-field needs Hall term for accuracy)
    |
    v
PIC Validation (V4: full discharge, needs V3 + absorbing BC)

Thomson Scattering (independent, needs Abel transform from interferometry.py)

AMR (Phase A-D, independent of PIC and Thomson)
    |
    v
PIC + AMR (future: PIC on AMR grid -- not in any scaffold)
```

### 6.2 Dependency Conflicts

| Conflict | Severity | Resolution |
|----------|----------|------------|
| PIC V3 E-field needs Hall MHD fix | MEDIUM | Hall H1 (30 LOC) must precede PIC V3. Not blocking V1-V2. |
| PIC V4 Yn validation needs absorbing BC | HIGH | BC fix (60 LOC) must precede V4, but scaffold lists it as MEDIUM. Promote to HIGH. |
| AMR + PIC on refined grid | FUTURE | Not addressed by any scaffold. Would require particle position mapping across AMR levels. Flag for Phase D+. |
| Thomson + two-temperature engine | LOW | Thomson reads Te from state dict. If 2T is not active, Te is derived from single-fluid T = p*m_i/(2*rho*kB). Works either way. |

### 6.3 Resource Conflicts

All four scaffolds can be worked on in parallel with two constraints:
1. **`mlx_sources.py`**: Both Hall MHD (H1-H3) and AMR (prolongation/restriction, if they touch source terms) edit this file. Hall MHD should go first (smaller scope, confirmed bug fix).
2. **`mlx_solver.py`**: Both Hall MHD (H4, wire CFL) and AMR (Phase A, wire AMR mode) edit this file. Sequential, not parallel.
3. **Test runtime budget**: PIC V4 (15 min), AMR Phase C conservation tests (~5 min), and full regression suite (~8 min) cannot all run in parallel on M3 Pro without memory pressure. Serialize slow tests.

### 6.4 Shared Infrastructure Opportunities

| Component | Used By | Status |
|-----------|---------|--------|
| Sub-cycling loop pattern | Hall MHD (H2), PIC (sub-cycling fix), AMR (Phase C subcycling) | Already exists as a pattern in resistive diffusion (`mlx_sources.py`). Extract as utility `operator_split_subcycle(fn, U, dt, dt_constraint, max_subcycles)` ~20 LOC. |
| Conservation audit | AMR (mass/energy tracking), PIC (V3 energy conservation) | Partial: `energy_balance.py` exists. Needs per-step delta tracking for AMR. ~30 LOC extension. |
| Abel transform | Thomson (spectral line integration), interferometry (existing) | Exists. Thomson reuses it. No new code needed. |
| Refinement sensors | AMR (Phase B), static_refinement.py (existing) | `lohner_error_indicator` and `current_density_sensor` confirmed at lines 321 and 380 of `static_refinement.py`. Reusable. |

---

## 7. Recommended Implementation Order

| Priority | Scaffold | Phase | Rationale |
|----------|----------|-------|-----------|
| 1 | Hall MHD | H1-H5 | Confirmed bug fix. Smallest scope (235 LOC). Unblocks PIC V3 accuracy. No dependencies. |
| 2 | Thomson Scattering | Full | Independent. Small scope (~200 LOC). High user-facing value (synthetic diagnostic). No physics coupling risk. |
| 3 | PIC Validation | V1-V2 | Unit tests + integration benchmarks. No coupling to other scaffolds. Establishes test baseline. |
| 4 | AMR | Phase A | Block data structures + ghost exchange. Largest scope, start early for calendar reasons. |
| 5 | PIC Validation | V3 | Requires Hall MHD (priority 1). MHD-coupled tests. |
| 6 | AMR | Phase B | Auto-refinement. Requires Phase A stabilization. |
| 7 | PIC Validation | V4 | Requires V3 + absorbing BC fix. End-to-end. Slow tests. |
| 8 | AMR | Phase C-D | Refluxing + subcycling. Research-grade. Defer until Phase A-B proven. |

Parallelism: Priorities 1+2 can run simultaneously. Priorities 3+4 can run simultaneously. Priority 5 waits on 1. Priority 7 waits on 5.

---

## 8. Missing Research Items

### Must-Answer Before Implementation

| # | Question | Scaffold | Blocking? |
|---|----------|----------|-----------|
| 1 | Does Parthenon-VIBE arXiv:2509.19701 actually exist? The ID prefix 2509 implies September 2025. If fabricated, the Amdahl analysis loses empirical grounding. | AMR | YES for Phase D |
| 2 | What is the optimal MLX block size for Apple M3 GPU? 32x64 exceeds L1 cache. Need profiling. | AMR | YES for Phase A |
| 3 | Is `scipy.special.wofz` sufficient for the plasma dispersion function at DPF-relevant alpha values (0.5-3)? Verify against Sheffield Table 5.1. | Thomson | YES for collective regime |
| 4 | What is the actual magnitude of E_Hall / E_conv at PF-1000 pinch conditions in the current MLX simulation? | PIC | YES for V3 |
| 5 | Does the existing `abel_transform` handle the singularity at r=y correctly for non-monotonic profiles (emissivity can have a ring structure)? | Thomson | MEDIUM |

### Nice-to-Have Before Implementation

| # | Question | Scaffold |
|---|----------|----------|
| 6 | Is there a Python implementation of cylindrical AMR refluxing anywhere (even research-quality) that could be adapted? | AMR |
| 7 | What particle count is needed for converged Yn in beam-target PIC? Schmidt (2012) used 10^6. | PIC |
| 8 | Can Nernst advection be comparable to Hall advection at DPF conditions? Quick order-of-magnitude check. | Hall MHD |
| 9 | Are there published Thomson scattering spectra from any DPF device beyond Decker (1996)? | Thomson |

---

## 9. Overall Readiness Scores

| Scaffold | Readiness | Confidence | Blocking Issues |
|----------|-----------|------------|-----------------|
| **Hall MHD** | **8/10** | HIGH | CFL formula bug in scaffold (ne_min vs ne_max); sub-cycle cap too low (20 vs needed 100+) |
| **Thomson Scattering** | **7/10** | HIGH | Missing plasma dispersion function approach; Te unit chain undocumented |
| **AMR** | **6/10** | MEDIUM | LOC underestimated 40-60%; no reference for cylindrical refluxing; unverified Parthenon-VIBE reference; block size not profiled |
| **PIC Validation** | **5/10** | MEDIUM | 3 blocking bugs (sub-cycling, E-field, reflecting BC); 8 total known bugs; cross-scaffold Hall dependency; particle count convergence unaddressed |

### Scoring Criteria
- 9-10: Ready to implement today. All research done, all dependencies met, estimates validated.
- 7-8: Implementation can start. Minor gaps addressable during work. No blocking unknowns.
- 5-6: Design is sound but significant research/prototyping needed before implementation. Some estimates are wishful.
- 3-4: Major gaps. Needs another design iteration.
- 1-2: Concept only. No implementable design.

---

## 10. Appendix: Wishful Thinking Flags

Items where the scaffolds assert something without evidence:

1. **AMR**: "Block-structured is the only viable option for MLX." -- True for `mx.compile()`, but the scaffold does not consider a hybrid approach: AMR bookkeeping in Python + compiled per-block kernels. This is what AMReX does. The "only viable" claim is overstrong.

2. **AMR**: "The Parthenon-VIBE Amdahl bottleneck is mitigable at 2 levels." -- Assertion without profiling data. The bottleneck may or may not exist at our scale. Need to measure.

3. **PIC**: "Order-of-magnitude agreement (10^10 - 10^12) is excellent for a first PIC-MHD hybrid." -- This is a 100x range. It is not "excellent"; it is "not broken." Reframe as: the first milestone is Yn > 0 from the correct mechanism (beam-target > thermonuclear).

4. **PIC**: "7-11 days of focused work." -- With 8 known bugs, 3 blocking, and production code changes required, 7 days is the optimistic bound. The realistic range is 12-18 days. The scaffold's own critical path diagram shows 7-11 days WITHOUT bug fix time.

5. **Hall MHD**: "Float32 is safe for Hall MHD at DPF conditions." -- The analysis covers single-step magnitudes but not sub-cycling accumulation over 10^7 steps. The conclusion is probably correct but the analysis is incomplete.

6. **Thomson**: "~50 LOC" for Salpeter correction. If using `scipy.special.wofz`, this drops to ~20 LOC. If implementing from scratch (no wofz), it is ~100 LOC. The estimate is in the middle of a bimodal distribution depending on an unresolved design choice.
