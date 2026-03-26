# Cycle 2: Deep Review of Cycle 1 Prototypes

**Date**: 2026-03-26 | **Reviewer**: dpf-validation-engineer (Opus 4.6)
**Method**: Six Sigma quality gate -- code review, estimate validation, dependency audit, contradiction detection

---

## 1. Per-Item Review Table

### Document 1: CYCLE1_PROTOTYPE_CODE.md (PIC V5 + Ghost Padding + Hall Validation)

| Item | Code OK? | Estimate OK? | Dependencies OK? | New Risks |
|------|----------|-------------|-------------------|-----------|
| **PIC V5 MLX full discharge** | PARTIAL -- `state.get("B")` returns wrong shape. MLX solver returns conserved array `(NVAR, nr, nz)`, not a dict with `"B"` key. NaN check iterates dict keys but `solver.step()` returns `mx.array`, not dict. | LOC OK (80). Time underestimated: MLX solver `step()` signature takes `(U, dt, current, voltage)` where U is an mx.array, not a state dict. Rewrite needed. | Depends on Esirkepov dt fix (correctly identified). Also depends on MLX solver returning state dict -- it does NOT. Major rewrite. | **HIGH**: The entire test is built around `state = dict` but MLX solver uses `mx.array` conserved state. Every assertion that accesses `state["rho"]`, `state.get("B")`, etc. will fail. |
| **Ghost padding GPU port** | GOOD -- Pure MLX ops, correct index arithmetic, proper energy consistency. The `mx.stack` reassembly is the right pattern. | LOC OK (100+30). Time realistic (2-3 hours). | Self-contained. No cross-item deps. | LOW: mx.where broadcast shape (RPN 90) is the top risk. Already has parity test. |
| **Hall mu_0 validation** | **CRITICAL ERROR** -- The prototype claims `E_Hall` needs a `mu_0` factor. This is WRONG. See Section 4 for full derivation. Implementing this "fix" would introduce a 1.26e-6x scaling bug. The whistler test is built to validate a bug that doesn't exist. | LOC: 20 fix + 60 test. The "fix" LOC is a negative -- it would break working code. Test LOC is valid but needs different acceptance criteria. | The Hall uniform-B no-op test (test 2) and density-scaling test (test 3) are correct and useful. Only the whistler phase speed test (test 1) has the wrong analytical target. | **CRITICAL**: RPN 300 risk item is based on a false premise. The mu_0 "fix" is the #1 risk to the project if implemented. |

### Document 2: CYCLE1_CALIBRATION_PROTOTYPE.md (Multi-Device Cal + Line Radiation + Species)

| Item | Code OK? | Estimate OK? | Dependencies OK? | New Risks |
|------|----------|-------------|-------------------|-----------|
| **Multi-device calibration sweep** | GOOD -- Imports verified, API matches. One bug: `CalibrationResult` lacks `nrmse` field (correctly identified in doc). The `getattr(..., 10.0)` workaround is fragile. | Wall time 96 min (serial) / 40 min (parallel) is realistic based on PF-1000 calibration experience (~24 min per device at 32x64). | Correctly notes Item 2 (line radiation) is optional. Can run bremsstrahlung-only. | MEDIUM: F1 (sim_time cap per device, RPN 168) is real. FAETON and UNU-ICTP sim_time defaults untested. |
| **Line radiation MLX** | GOOD -- Log-space arithmetic follows bremsstrahlung pattern. Cu 6-segment approximation is reasonable for DPF temperatures. Energy clamping correct. | LOC OK (100). The 6-segment Cu table is a smart simplification vs 21-point full table. | Needs `SpeciesManager` (exists) and `species_Y` array. Integration point in mlx_solver.py is clearly specified. | LOW: F3 (float32 subnormals, RPN 224) is correctly mitigated by log-space. Minor: `U[10-1]` hardcodes IEE index instead of using the constant. |
| **Multi-species E2E test** | **PARTIAL** -- Test 1 never calls `species_advection_step` in the loop, so species Y stays constant. The doc's "Known Gap" says "the solver doesn't drive species advection internally" -- this is WRONG. `mlx_solver.py:760` already calls `species_advection_step`. The test fails to exercise existing wiring. Test 2 is correct and useful. | LOC OK (60). | **FALSE DEPENDENCY CLAIM**: Doc says engine wiring needed for species advection. It's already wired at `mlx_solver.py:754-763`. The test should use the solver's built-in species support, not bypass it. | MEDIUM: Test 1 will pass but prove nothing about species advection because it doesn't use the solver's species integration path. |

### Document 3: CYCLE1_INTEGRATION_PROTOTYPE.md (AMR + Thomson UI + Differentiable MHD)

| Item | Code OK? | Estimate OK? | Dependencies OK? | New Risks |
|------|----------|-------------|-------------------|-----------|
| **AMR integration test** | PARTIAL -- `MLXMHDSolver(config_amr)` assumes the solver accepts a `SimulationConfig` object. Current constructor takes keyword args (`grid_shape`, `dx`, `dz`, etc.), not a config object. `_total_mass` sums rho without volume weighting (cylindrical cells have volume `2*pi*r*dr*dz`) -- conservation assertion is wrong. | Time OK (2 hours write + 30 min run). LOC OK. But `_measure_sheath_width` accesses `state["B"]` which is not the MLX solver's output format (same dict-vs-array issue as PIC V5). | Correctly identifies `_step_amr` rhs_fn=None (RPN 80) and CF ghost exchange gap (RPN 128). | HIGH: Volume-weighting bug in `_total_mass` means the mass conservation assertion passes/fails for wrong reasons. On a cylindrical grid, `sum(rho)` is NOT mass. |
| **Thomson Gradio UI** | GOOD -- Clean separation of compute and UI. `matplotlib.use("Agg")` is correct. Error handling for None state is present. | LOC OK (100). Time OK (3 hours). | Correctly identifies Lee model guard (RPN 96) and Te unit mismatch (RPN 84). | LOW: `velocity[1, :, iz]` should be `velocity[2, :, iz]` for z-component (index 1 is z in our state dict but the code comment says "vz" -- needs verification against actual state dict layout). |
| **Differentiable MHD research brief** | N/A (research, not code) -- The audit of MLX ops is thorough and correct. `mx.where` gradient behavior is correctly flagged as the key unknown. | 3-4 days prototype is realistic. The 500 LOC total is reasonable. | `hlls_flux_r` function does not exist in the codebase. The smoke test imports `from dpf.metal.mlx_riemann import hlls_flux_r` which will fail. The actual function names need verification. | LOW: Research brief quality is high. The `mx.array.at[].add()` syntax in the smoke test is JAX-style, not MLX. MLX uses `U.at[0, ...].set(value)` or indexing assignment. |

---

## 2. Cross-Document Contradictions

| # | Contradiction | Documents | Severity | Resolution |
|---|-------------|-----------|----------|------------|
| C1 | **Hall mu_0 factor**: Prototype doc says "E_Hall needs mu_0 * (J x B)/(ne*e)" (line 386). Scaffold review (SCAFFOLD_REVIEW_SIX_SIGMA.md Section 4.1) agrees. But the actual code is CORRECT: in HL units, J = curl(B) and E_Hall = (J x B)/(ne*e) with no mu_0 factor. See Section 4 for derivation. | CYCLE1_PROTOTYPE_CODE.md vs actual physics | **CRITICAL** | Do NOT implement the mu_0 "fix". It would break the Hall term. |
| C2 | **Species advection wiring**: Calibration doc (line 655-660) says "the solver doesn't drive species advection internally -- it must be called explicitly." But `mlx_solver.py:754-763` already wires `species_advection_step` into the solver loop. | CYCLE1_CALIBRATION_PROTOTYPE.md vs codebase | HIGH | Update test to use solver's built-in species path, not manual calls. |
| C3 | **MLX solver API**: PIC V5 and AMR tests treat the solver as returning a state dict (`state["rho"]`, `state.get("B")`). The MLX solver `step()` returns an `mx.array` of shape `(NVAR, nr, nz)`. There IS a `get_state()` method that returns a dict, but it's a separate call. | CYCLE1_PROTOTYPE_CODE.md, CYCLE1_INTEGRATION_PROTOTYPE.md vs codebase | HIGH | All tests using dict access on solver output need rewriting. Use `solver.get_state()` after `solver.step()`, or index the conserved array directly. |
| C4 | **Risk register vs prototype FMEA**: Risk plan R03 (score 20) and Cycle 1 FMEA item #1 (RPN 300) both claim Hall mu_0 is a confirmed bug. Based on C1, these entries are wrong. | RISK_MANAGEMENT_PLAN.md, CYCLE1_PROTOTYPE_CODE.md | HIGH | Remove R03 from RED risk list. Hall may have OTHER bugs, but mu_0 is not one of them. |
| C5 | **HLLS function name**: Differentiable MHD test imports `hlls_flux_r` but no such function exists in `src/dpf/metal/`. The actual Riemann solver functions need verification. | CYCLE1_INTEGRATION_PROTOTYPE.md vs codebase | LOW | Find actual function name before implementing smoke test. |

---

## 3. Revised Implementation Priority

Original order across all 3 docs: Hall mu_0 fix -> Ghost padding -> PIC V5 -> Line radiation -> Species test -> Multi-device cal -> AMR test -> Thomson UI -> Differentiable MHD.

**Revised order** (incorporating findings):

| Priority | Item | Rationale | Change from Cycle 1 |
|----------|------|-----------|---------------------|
| 1 | **Ghost padding GPU port** (Doc 1, Item 2) | Self-contained, no API confusion, clear parity test. Delivers 5-8% perf gain. | Promoted from #2 to #1 |
| 2 | **Hall validation tests** (Doc 1, Item 3 -- tests only, NO mu_0 fix) | The uniform-B no-op test and density-scaling test are valid. Remove the whistler phase speed target based on the wrong mu_0 claim. Write tests that validate the CURRENT code is correct. | RADICALLY changed: was "fix mu_0 first" (RPN 300), now "validate current code is correct" |
| 3 | **Line radiation MLX** (Doc 2, Item 2) | Clean implementation, follows existing bremsstrahlung pattern. | Unchanged |
| 4 | **Multi-species E2E test** (Doc 2, Item 3) | Rewrite test 1 to use solver's built-in species path (already wired). Remove false dependency claim. | Rewritten to use existing wiring |
| 5 | **Multi-device calibration** (Doc 2, Item 1) | Can start with bremsstrahlung-only. Add sim_time caps per device. | Unchanged |
| 6 | **Thomson UI** (Doc 3, Item 2) | Independent, low risk, high user-facing value. | Unchanged |
| 7 | **PIC V5** (Doc 1, Item 1) | Needs full rewrite to use MLX solver's actual API (mx.array, not dict). Defer until API is clarified. | Demoted from #3 to #7 |
| 8 | **AMR integration test** (Doc 3, Item 1) | Needs volume-weighting fix, solver API fix, and the existing `_step_amr` source-term gap. | Demoted; more rework needed than estimated |
| 9 | **Differentiable MHD** (Doc 3, Item 3) | Research-only. Run smoke test when function names are verified. | Unchanged |

---

## 4. The Hall mu_0 Verdict: FALSE ALARM

### Claim (Cycle 1 Prototype Doc, line 386)

> "Current code: `E_Hall = (J_HL x B_HL) / (n_e * e)`. Correct HL: `E_Hall = mu_0 * (J_HL x B_HL) / (n_e * e)`. Without mu_0, Hall is ~10^6 too weak."

### Derivation

**In SI units**:
- `J_SI = curl(B_SI) / mu_0`
- `E_Hall_SI = (J_SI x B_SI) / (n_e * e)`

**In HL units** (where `B_HL = B_SI / sqrt(mu_0)`):
- `curl(B_HL) = curl(B_SI) / sqrt(mu_0) = J_SI * mu_0 / sqrt(mu_0) = J_SI * sqrt(mu_0)`
- The code sets `J_HL = curl(B_HL)`, so `J_HL = J_SI * sqrt(mu_0)`
- `J_HL x B_HL = J_SI * sqrt(mu_0) * B_SI / sqrt(mu_0) = J_SI * B_SI`
- `E_Hall = (J_HL x B_HL) / (n_e * e) = (J_SI * B_SI) / (n_e * e) = E_Hall_SI`

**Result**: The current code `E_Hall = (J x B) / (n_e * e)` is **CORRECT** in HL units. No mu_0 factor is needed.

### What went wrong in Cycle 1

The prototype doc (and the scaffold review, and the risk register) all assumed that `J_HL = curl(B_HL)` differs from `J_SI` by a factor of mu_0. In fact, it differs by `sqrt(mu_0)`, which is exactly compensated by the `sqrt(mu_0)` factor in `B_HL`. The cross product `J x B` has identical magnitude in SI and HL units.

The earlier fix (removing the erroneous `/MU_0` in `compute_current_density_components`) was correct and sufficient. No further mu_0 factor is needed anywhere in the Hall computation.

### Impact

- **R03 in RISK_MANAGEMENT_PLAN.md** (score 20, RED): Downgrade to GREEN. The Hall term is physically correct. It may still be numerically weak (due to small perturbations or resolution), but not by a factor of 10^6.
- **FMEA Item 1** in Hall validation (RPN 300): Remove. The "highest risk" item is a false positive.
- **Whistler dispersion test**: The analytical target is still valid, but the expected behavior is that the CURRENT code should PASS (not fail). If it fails, the issue is numerical (resolution, boundary artifacts), not a missing mu_0.
- **Cycle 1 implementation order**: "Fix mu_0 BEFORE running validation" is wrong. Run validation FIRST to confirm the code is correct.

### Recommended action

Write the Hall validation tests with the understanding that E_Hall = (J x B)/(ne*e) is correct. If the whistler test shows the Hall term is too weak, investigate numerical causes (resolution, boundary wrap-around, NaN clamping masking real cells) rather than inserting a mu_0 factor.

---

## 5. Updated FMEA (New Risks from Cycle 2)

| # | Failure Mode | Sev | Occ | Det | RPN | Source | Mitigation |
|---|-------------|-----|-----|-----|-----|--------|------------|
| N1 | **False mu_0 "fix" breaks Hall MHD** -- implementing the Cycle 1 recommendation multiplies E_Hall by 1.26e-6, making Hall 10^6 too STRONG (inverse of claimed error) | 10 | 7 (3 docs recommend it) | 2 (tests would catch) | 140 | Cycle 2 review | Do NOT implement. Run validation tests first. Update all 3 docs. |
| N2 | **PIC V5 / AMR tests use wrong solver API** -- dict access on mx.array causes TypeError at step 0 | 8 | 10 (code is written) | 1 (immediate crash) | 80 | Cycle 2 code review | Rewrite tests to use `solver.get_state()` or conserved array indexing. |
| N3 | **AMR mass conservation test lacks volume weighting** -- `sum(rho)` != mass on cylindrical grid, assertion passes/fails for wrong reason | 7 | 10 (code is written) | 6 (subtle: test might still "pass") | 420 | Cycle 2 code review | Replace with `sum(rho * 2*pi*r*dr*dz)` per cell. |
| N4 | **Species E2E test bypasses existing wiring** -- test manually manages Y instead of using solver's built-in species_advection_step | 5 | 10 (code is written) | 4 (test passes but proves nothing about integration) | 200 | Cycle 2 contradiction check | Rewrite test to construct solver with species config and let solver manage Y internally. |
| N5 | **Differentiable MHD smoke test uses JAX API** -- `QL.at[0].add(rho_L_val)` is JAX syntax, not MLX | 4 | 10 (code is written) | 1 (immediate AttributeError) | 40 | Cycle 2 code review | Use MLX indexing: `QL = QL.at[0, ...].set(QL[0] + rho_L_val)` or build array directly. |

### Updated Top 5 Risks (replacing Cycle 1 rankings)

| Rank | ID | RPN | Description |
|------|-----|-----|-------------|
| 1 | N3 | 420 | AMR mass conservation test has wrong metric (no volume weighting) |
| 2 | N4 | 200 | Species E2E test bypasses existing solver wiring |
| 3 | F3 (Cal doc) | 224 | Float32 subnormals in Cu line radiation (mitigated by log-space) |
| 4 | F4 (Cal doc) | 180 | Vacuum Z_eff catastrophe (mitigated by existing mask) |
| 5 | N1 | 140 | False mu_0 "fix" would break Hall MHD if implemented |

Removed from top risks:
- Hall mu_0 (was RPN 300) -- false alarm, see Section 4
- Esirkepov dt mismatch (was RPN 252) -- still valid but not changed

---

## 6. Specific Corrections Needed in Cycle 1 Docs Before Implementation

### CYCLE1_PROTOTYPE_CODE.md

| Line | Issue | Correction |
|------|-------|------------|
| 386-387 | "Correct HL: E_Hall = mu_0 * (J_HL x B_HL) / (n_e * e)" | DELETE. Current code is correct. Replace with: "Current code is correct in HL units. Validate via whistler dispersion test." |
| 98-104 | `state = solver.step(...)` used as dict | MLX solver returns `mx.array`. Use `solver.step()` then `solver.get_state()` for dict access. |
| 124-128 | `state.get("B")` | Does not exist on mx.array. Extract B from conserved array: `B = state_arr[IBR:IBT+1]` |
| 605 (FMEA) | RPN 300 for mu_0 factor | Reduce to RPN 0. Not a real risk. Replace with: "Validate Hall magnitude via density-scaling test." |
| 637 | Implementation priority "3 -> 2 -> 1" based on mu_0 RPN 300 | Reverse: "2 -> 3 (tests only) -> 1". Ghost padding first, Hall validation second (no fix needed), PIC last. |

### CYCLE1_CALIBRATION_PROTOTYPE.md

| Line | Issue | Correction |
|------|-------|------------|
| 145 | `getattr(cal_result, "nrmse", 10.0)` | Replace with extracting nrmse from the best trial object. The doc already identifies this bug -- just ensure it's fixed before implementation. |
| 435 | `U[10 - 1]` hardcodes IEE index | Use `U[IEE]` with import from mlx_kernels. |
| 655-660 | "the solver doesn't drive species advection internally" | WRONG. `mlx_solver.py:754-763` already calls `species_advection_step`. Rewrite test to use solver's built-in path. |
| 536 | `Y.shape == (1, nr, nz)` | Verify: for D+Cu with D as background, only Cu is evolved, so shape is (1, nr, nz). Correct. |

### CYCLE1_INTEGRATION_PROTOTYPE.md

| Line | Issue | Correction |
|------|-------|------------|
| 85-88 | `_total_mass` uses `sum(rho)` without volume weighting | Must be `sum(rho * 2*pi*r*dr*dz)` for cylindrical grid. Without this, the conservation assertion is meaningless. |
| 100 | `MLXMHDSolver(config_amr)` | Solver takes keyword args, not SimulationConfig. Rewrite construction. |
| 124-125 | `state_amr["B"]` etc. | Same dict-vs-array issue. Use `solver.get_state()`. |
| 439 | `hlls_flux_r` import | Function name does not exist in codebase. Find actual Riemann function name. |
| 449 | `QL.at[0].add(rho_L_val)` | JAX syntax, not MLX. Rewrite. |

### RISK_MANAGEMENT_PLAN.md

| ID | Issue | Correction |
|----|-------|------------|
| R03 | "Hall MHD unit bug (mu_0 missing)" scored 20 (RED) | Downgrade to GREEN (score 4). Hall E_Hall computation is correct in HL units. The earlier MU_0 removal fix was sufficient. |
| R03 mitigation | "Fix in H1 (30 LOC)" | Change to: "Validate via whistler dispersion test. No code fix needed for mu_0." |

### SCAFFOLD_REVIEW_SIX_SIGMA.md

| Section | Issue | Correction |
|---------|-------|------------|
| 4.1 Measure | "The HL unit derivation in Section 2 is correct... E_H = (J_HL x B_HL) * mu_0 / (n_e * e)" | WRONG. The derivation in Section 2 of the Hall design doc has an error. Correct result: E_H = (J_HL x B_HL) / (n_e * e), no mu_0 factor. |
| 4.2 FMEA #3 | "Hall inactive in production (mu_0 error = 10^6 wrong)" RPN 480 | Hall IS active at correct magnitude. Reduce RPN to 0. |

---

## Summary

**Cycle 1 delivered 9 prototype items of mixed quality:**
- 3 items are implementation-ready with minor fixes (ghost padding, line radiation, Thomson UI)
- 3 items need API corrections (PIC V5, AMR test, differentiable MHD smoke test)
- 2 items have false premises that would cause bugs if implemented (Hall mu_0 "fix", species wiring "gap")
- 1 item is solid research (differentiable MHD brief)

**The single most important finding**: The Hall mu_0 claim is wrong. Three documents (prototype, scaffold review, risk plan) independently recommended a "fix" that would multiply E_Hall by mu_0 = 1.26e-6, making the Hall term 10^6 too STRONG -- the exact opposite of the claimed error. This is a textbook case of a multi-document confirmation bias where an initial error propagated through the review chain without independent verification against the actual unit algebra.

**Action required before Cycle 3**: Correct all mu_0 references across the 5 documents. Rewrite PIC V5 and AMR tests to use MLX solver's actual API. Update species test to use existing wiring.
