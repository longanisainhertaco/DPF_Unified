# AMR Phase B: Automatic Refinement — Implementation Spec

**Date**: 2026-03-26  |  **Budget**: ~350 LOC prod + ~200 LOC tests
**Prereqs**: `AMR_PHASE_A_IMPL_SPEC.md`, `AMR_PROTOTYPE_CODE.md`, `amr_concerns_analysis.md`
**Refs**: Keppens et al. MPI-AMRVAC 3.0 A&A 673:A66 (2023), Stone et al. ApJS 249:4 (2020),
Lohner Comp.Meth.Appl.Mech.Eng 61:323 (1987)

---

## 1. Prerequisites from Phase A

All of the following must be true before Phase B begins. Verify with the listed check.

| # | Prerequisite | Verification |
|---|-------------|--------------|
| 1 | `AMRHierarchy`, `AMRLevel`, `AMRBlock` dataclasses exist | `from dpf.metal.mlx_amr import AMRHierarchy` succeeds |
| 2 | `amr_step()` orchestrator exists and handles global-timestep stepping | `pytest tests/test_mlx_amr.py::test_amr_step_uniform_preserved -x` passes |
| 3 | `build_amr_hierarchy(refined_blocks=None)` creates a 2-level hierarchy with empty level 1 | `pytest tests/test_mlx_amr.py::test_build_hierarchy_manual -x` passes |
| 4 | `prolongate_to_fine()` and `restrict_to_coarse()` both verified | Tests 7-10 in Phase A suite pass |
| 5 | `AMRConfig` in `config.py` with all 11 fields including `regrid_interval`, thresholds, `buffer_width` | `AMRConfig().regrid_interval == 20` |
| 6 | All 4,861 existing tests pass | `pytest tests/ -x -q -m "not slow"` green |
| 7 | `lohner_error_indicator` and `current_density_sensor` in `static_refinement.py` at lines 321-425 | `from dpf.experimental.static_refinement import lohner_error_indicator` succeeds |

If any prerequisite fails, **do not begin Phase B**. Fix Phase A first.

---

## 2. New Functions

All new code lives in `src/dpf/metal/mlx_amr.py` as additions to the Phase A module.
Total new LOC: ~350 production, ~200 tests.

### 2.1 `evaluate_refinement_sensors`

```python
def evaluate_refinement_sensors(
    hierarchy: AMRHierarchy,
) -> dict[tuple[int, int, int], tuple[float, float]]:
```

**Purpose**: Run Lohner (density) and current-density sensors on every active leaf block.
Returns `{(level_idx, ir, iz): (j_val, lohner_val)}`.

**Algorithm** (~35 LOC):
1. Iterate over all `(li, block)` pairs. Skip blocks with `block.active == False`.
2. For each block: `U_np = np.array(block.U)` (pull from MLX to CPU; sensors are NumPy).
3. `rho = U_np[IDN]`; `B = np.stack([U_np[IBR], U_np[IBZ], U_np[IBT]])`.
4. Call `lohner_indicator_block(rho, level.dr, level.dz)` — the 2D version from `AMR_PROTOTYPE_CODE.md:47-73`. Returns scalar max.
5. Call `current_density_sensor_block(B, level.dr, level.dz)` — from `AMR_PROTOTYPE_CODE.md:76-89`. Returns scalar max.
6. Store `result[(li, ir, iz)] = (j_val, l_val)`.
7. Return dict.

**Note**: Do NOT call the 3D `lohner_error_indicator` from `static_refinement.py`
directly — it expects a 3D `(nr, ny, nz)` array with a y-dimension that blocks
don't have. The 2D `lohner_indicator_block` from the prototype is the correct
version. Copy it into `mlx_amr.py` as a private helper.

**LOC estimate**: 35

---

### 2.2 `flag_blocks_for_refinement`

```python
def flag_blocks_for_refinement(
    sensor_values: dict[tuple[int, int, int], tuple[float, float]],
    config: AMRConfig,
) -> dict[tuple[int, int, int], int]:
```

**Purpose**: Convert sensor scalars to flags: `+1` (refine), `-1` (derefine), `0` (keep).
Applies hysteresis: refine threshold > derefine threshold prevents oscillation.

**Algorithm** (~20 LOC):
1. For each key `(li, ir, iz)` in sensor_values:
   - `j_val, l_val = sensor_values[key]`
   - If `li < config.max_levels - 1` AND (`j_val > config.j_threshold_refine` OR `l_val > config.lohner_threshold_refine`): flag `+1`
   - Elif `li > 0` AND `j_val < config.j_threshold_derefine` AND `l_val < config.lohner_threshold_derefine`: flag `-1`
   - Else: flag `0`
2. Return flags.

**Hysteresis gap**: `refine_thresh / derefine_thresh = 6x` (J: 0.3/0.05, Lohner: 0.2/0.03).
Derived from MPI-AMRVAC defaults (Keppens 2023 Section 4.2): ratio >= 5x prevents
flag oscillation when sensor value straddles the threshold due to numerical noise.
Athena++ uses a fixed 10:1 ratio for its `deref_threshold = 0.1 * ref_threshold`.
DPF sheath sensors are noisier than Athena++ linear wave tests, so 6:1 is a
conservative midpoint.

**LOC estimate**: 20

---

### 2.3 `enforce_proper_nesting`

```python
def enforce_proper_nesting(
    flags: dict[tuple[int, int, int], int],
    hierarchy: AMRHierarchy,
    config: AMRConfig,
) -> dict[tuple[int, int, int], int]:
```

**Purpose**: Block AMR requires no fine block without a coarse neighbor (proper nesting).
Also implements buffer zones: any block neighboring a `+1` block must become `+1` or
at minimum `0`. Two passes:

**Algorithm** (~40 LOC):
- **Pass 1 — buffer zone expansion**:
  For each `(li, ir, iz)` with flag `+1`: for all neighbors within `config.buffer_width`
  blocks in (r,z), if their flag is `-1` or `0`, promote to `+1`.
  Buffer width default is 1 (one coarse block). This ensures the sheath region has
  refined blocks on both sides, preventing ghost-cell errors at the sheath boundary.

- **Pass 2 — proper nesting**:
  For each `(li, ir, iz)` with flag `+1` where `li > 0`: check that parent block
  `(li-1, ir//ratio, iz//ratio)` exists in `hierarchy.levels[li-1].blocks`. If not,
  downgrade flag to `0`. This prevents orphan fine blocks.

- **Pass 3 — capacity cap**:
  Count current active fine blocks + proposed new `+1` blocks. If total would exceed
  `config.max_blocks_per_level`, sort candidates by max(j_val, l_val) descending and
  keep only the top-N. This prevents runaway refinement.

**LOC estimate**: 40

---

### 2.4 `create_child_blocks`

```python
def create_child_blocks(
    hierarchy: AMRHierarchy,
    parent_block: AMRBlock,
    config: AMRConfig,
) -> list[AMRBlock]:
```

**Purpose**: Prolongate a coarse block to `ratio^2` fine children. Creates AMRBlock
objects with correct r_min, z_min, and prolongated U.

**Algorithm** (~35 LOC):
1. Ensure `hierarchy.levels` has a level `parent_block.level + 1`. If not, call
   `hierarchy.add_level(dr=level.dr/ratio, dz=level.dz/ratio)`.
2. `U_np = np.array(parent_block.U)` — shape `(NVAR, block_nr, block_nz)`.
3. For `di in range(ratio)`, `dj in range(ratio)`:
   - Extract quadrant: `quad = U_np[:, di*hnr:(di+1)*hnr, dj*hnz:(dj+1)*hnz]` where
     `hnr = block_nr//ratio`, `hnz = block_nz//ratio`.
   - Call `_prolongate_vanleer(quad, ratio)` — from `AMR_PROTOTYPE_CODE.md:215-238`.
     Returns `(NVAR, block_nr, block_nz)` fine block.
   - Compute `cidx = (parent_block.index[0]*ratio + di, parent_block.index[1]*ratio + dj)`.
   - Compute `r_min = parent_block.r_min + di * block_nr * fine_dr`.
   - Compute `z_min = parent_block.z_min + dj * block_nz * fine_dz`.
   - Create `AMRBlock(level=fi, index=cidx, U=mx.array(U_fine), r_min=..., z_min=..., active=True)`.
   - If `cidx` already exists in `fine.blocks`, skip (block already refined from earlier regrid).
4. Return list of new children. Also store in `fine.blocks`.

**LOC estimate**: 35

---

### 2.5 `remove_child_blocks`

```python
def remove_child_blocks(
    hierarchy: AMRHierarchy,
    child_block: AMRBlock,
    config: AMRConfig,
) -> None:
```

**Purpose**: Restrict a fine block back to coarse parent, then deactivate it.

**Algorithm** (~20 LOC):
1. `li = child_block.level`. Assert `li > 0`.
2. Get parent: `pidx = (child_block.index[0] // ratio, child_block.index[1] // ratio)`.
   `parent = hierarchy.levels[li-1].blocks.get(pidx)`.
3. If parent exists: call `restrict_to_coarse([child_block], parent, hierarchy.levels[li], ratio, block_nr, block_nz)`.
   This is the Phase A function — call it here to update parent before removing child.
4. `del hierarchy.levels[li].blocks[child_block.index]`.
5. If `hierarchy.levels[li].blocks` is empty after removal, **do not remove the level** —
   the level container stays so subsequent regrids can repopulate it without re-adding.

**LOC estimate**: 20

---

### 2.6 `auto_regrid`

```python
def auto_regrid(
    hierarchy: AMRHierarchy,
    config: AMRConfig,
) -> tuple[AMRHierarchy, int, int]:
```

**Purpose**: Full regrid orchestrator. Evaluate sensors → flag → nest → create/remove.
Returns `(updated_hierarchy, n_refined, n_derefined)` for diagnostics.

**Algorithm** (~50 LOC):
1. `sensor_values = evaluate_refinement_sensors(hierarchy)` — ~35 LOC function above.
2. `flags = flag_blocks_for_refinement(sensor_values, config)` — ~20 LOC.
3. `flags = enforce_proper_nesting(flags, hierarchy, config)` — ~40 LOC.
4. `n_refined = 0; n_derefined = 0`
5. **Refine pass**: for each `(li, ir, iz)` with flag `+1`:
   - parent = `hierarchy.levels[li].blocks.get((ir, iz))`. Skip if None.
   - children = `create_child_blocks(hierarchy, parent, config)`.
   - `n_refined += len(children)`
6. **Derefine pass**: for each `(li, ir, iz)` with flag `-1`:
   - block = `hierarchy.levels[li].blocks.get((ir, iz))`. Skip if None.
   - Check all `ratio^2` sibling blocks also have flag `-1`. If any sibling is `0` or `+1`,
     skip (can only derefine a complete set of siblings).
   - `remove_child_blocks(hierarchy, block, config)`.
   - `n_derefined += 1`
7. If `n_refined > 0` or `n_derefined > 0`: call `hierarchy.fill_all_ghosts()`.
8. Return `(hierarchy, n_refined, n_derefined)`.

**LOC estimate**: 50

---

## 3. Integration into `amr_step()`

Phase A's `amr_step()` signature (from `AMR_PHASE_A_IMPL_SPEC.md:139`):
```python
def amr_step(hierarchy, dt, gamma, method, riemann, ng, current, r_inner,
             step_number, rhs_fn, use_refluxing=True) -> (AMRHierarchy, float):
```

**Where to insert `auto_regrid`**: At the **top** of `amr_step()`, before ghost exchange
and before the RHS. Regrid at the start of the step so that the new block topology is
established before any flux computation. This matches Athena++ and MPI-AMRVAC timing.

**Exact insertion** (3 lines):
```python
def amr_step(hierarchy, dt, ..., step_number, rhs_fn, use_refluxing=True, config=None):
    if config is not None and step_number % config.regrid_interval == 0 and step_number > 0:
        hierarchy, _, _ = auto_regrid(hierarchy, config)
    # ... rest of Phase A unchanged
```

**Why before RHS, not after**: Regriding after the RHS would leave newly created fine
blocks in an unstepped state for one step. Regriding before means all blocks (new and
old) get the same RHS treatment in the current step. Athena++ (`mesh.cpp:MeshRefinement()`)
does this before the time driver loop. MPI-AMRVAC does it at step boundary.

**Why `step_number > 0`**: Skip step 0 — the hierarchy was just built from `build_amr_hierarchy()`.
Regriding immediately would discard manually-specified `refined_blocks`.

---

## 4. Testing Plan

All tests in `tests/test_mlx_amr.py`, appended to Phase A test suite. Gate on `pytest.importorskip("mlx.core")`.

| # | Test | Inputs | Expected Output |
|---|------|--------|-----------------|
| B1 | `test_sensor_fires_on_sheath` | 2-block hierarchy; inject a sharp density gradient (factor-10 jump) in block (0,0,0) | `sensor_values[(0,0,0)][1] > 0.5`; block (0,1,0) sensor < 0.1 |
| B2 | `test_flag_hysteresis` | Sensor value sweep from 0.0 to 1.0 in 0.01 steps | Flag `+1` only above 0.3; flag `-1` only below 0.05; `0` in between. No oscillation. |
| B3 | `test_proper_nesting_rejects_orphan` | Flag level-1 block for refine but level-0 parent absent | `enforce_proper_nesting` downgrades flag to `0` |
| B4 | `test_create_children_count_and_position` | Coarse block at (0, 1, 2), ratio=2 | 4 children at (1, 2, 4), (1, 2, 5), (1, 3, 4), (1, 3, 5) with correct r_min/z_min |
| B5 | `test_create_children_mass_conservation` | Coarse block with known `rho` profile | `sum(child rho * fine_vol)` == `sum(coarse rho * coarse_vol)` to 1e-5 relative |
| B6 | `test_remove_child_restricts_to_parent` | Fine block with modified `rho`; call `remove_child_blocks` | Parent `rho` updated; child no longer in `hierarchy.levels[1].blocks` |
| B7 | `test_auto_regrid_uniform_no_change` | Hierarchy with uniform `rho=1.0`; call `auto_regrid` | `n_refined == 0`, `n_derefined == 0`, hierarchy unchanged |
| B8 | `test_auto_regrid_shock_triggers_refinement` | Hierarchy with 10:1 density jump in one block | `n_refined == 4` (one parent -> 4 children); children exist in `hierarchy.levels[1]` |
| B9 | `test_auto_regrid_called_at_interval` | `amr_step` with `config.regrid_interval=5`; run 10 steps | `auto_regrid` called exactly once (at step 5); confirmed via mock or counter |
| B10 | `test_amr_step_mass_conservation_with_regrid` | Sod-like problem across blocks; run 40 steps (2 regrids); check mass | `|dm/m| < 1e-4` per step with refluxing active |

---

## 5. Hysteresis Tuning for DPF Sheath Tracking

### Recommended defaults

```python
j_threshold_refine:    0.30   # ~30% of max normalized J
j_threshold_derefine:  0.05   # ~5% — 6x hysteresis gap
lohner_threshold_refine:    0.20
lohner_threshold_derefine:  0.03   # ~6.7x hysteresis gap
buffer_width: 1                # one block on each side of sheath
```

### Derivation

**MPI-AMRVAC defaults** (Keppens 2023, `amrvac.par` example files): `refine_threshold=0.3`,
no explicit derefine threshold in base docs — estimated at `0.1 * refine_threshold` from
source (`src/mod_amr_fct.t`, function `forcedrefine`). That gives a 10:1 ratio.

**Athena++ defaults** (Stone et al. 2020, `parameter_input.cpp`): `RefineLevel1=0.5` for
density gradient; no deref threshold in base code — Parthenon sets it to `0.25 * refine`.
4:1 ratio.

**DPF-specific calibration** (from amr_concerns_analysis.md, Concern 1):
The DPF sheath sensor is higher-noise than Athena++ test problems because:
- Cylindrical geometry amplifies radial gradients at small r.
- Two-temperature plasma creates T_e/T_i discontinuities that trigger Lohner on non-sheath features.
- Pinch phase has sharp pressure spikes (factor 100x over fill) not present in standard benchmarks.

A 6:1 ratio prevents oscillation under these conditions. The 0.30 refine threshold
is validated against the DPF sheath sensor from `static_refinement.py` line 418, which
normalizes by `B_max` and peaks to 0.6-0.8 at the sheath during axial rundown.

**buffer_width=1** (one block): follows Athena++ `ncell_refine_buffer=1` default.
MPI-AMRVAC uses `refine_max_level=2` with implicit 1-block buffer. For DPF, the
sheath moves ~0.04 mm per step, so it takes ~375 steps to cross one 16-cell block
at default resolution. Buffer of 1 block means the sensor fires with ~375-step
advance warning — ample lead time.

---

## 6. Performance Budget

### Regrid cost target

Regrid runs every `config.regrid_interval = 20` steps. At 20,000 steps total
(8 us PF-1000 discharge), that is 1,000 regrid calls.

Target: each `auto_regrid` call < 1 ms wall-clock.

Breakdown:
- `evaluate_refinement_sensors`: 4-8 blocks * NumPy sensor computation on (16,32) array.
  Sensor cost per block: ~0.1 ms (16*32*10 = 5,120 floating-point ops, vectorized NumPy).
  Total: 8 blocks * 0.1 ms = 0.8 ms.
- `flag_blocks_for_refinement` + `enforce_proper_nesting`: pure Python dict operations,
  O(N_blocks). N=8: < 0.01 ms.
- `create_child_blocks` / `remove_child_blocks`: prolongation on (10, 16, 32) -> (10, 16, 32).
  Van Leer slope computation: ~0.2 ms per parent, typically 0-2 parents per regrid.
  Total: < 0.4 ms.
- `hierarchy.fill_all_ghosts()`: ~0.015 ms per block (from concerns analysis Concern 2: 14 us per full step / 3 stages ≈ 5 us, but full ghost fill without RHS is cheaper). With 8-12 blocks: ~0.1 ms.

**Total estimated `auto_regrid`: ~1.3 ms worst case, ~0.3 ms typical.**

At 20-step cadence:
- Regrid cost per amr_step: 1.3 ms / 20 steps = 0.065 ms/step overhead.
- Base amr_step cost (from concerns Concern 3): ~0.625 ms/step at 12,288 cells.
- **Regrid fraction: 10% worst case, 3% typical.**

If regrid cost exceeds 5 ms (measured), increase `regrid_interval` to 50 and
reduce block sensor evaluation from full-array NumPy to sampled (every 4th cell).

### Runtime fraction at every-20-steps cadence

Total regrid time: 1,000 calls * 1.3 ms = 1.3 seconds.
Total simulation time at 12.5 min (f=12.5% refined): 750 seconds.
**Regrid fraction of total runtime: 0.17%.** Negligible.

---

## 7. Risk Assessment

| # | Failure Mode | Probability | Severity | Mitigation |
|---|-------------|-------------|----------|------------|
| R1 | **Over-refinement cascade**: sensor fires on numerical artifacts (axis singularity, vacuum boundary), entire domain gets refined, simulation becomes 8x slower | Medium (the axis at r=0 always has high J due to 1/r) | High | Mask first 2 radial blocks from refinement (`ir <= 1` → force flag 0); cap at `max_blocks_per_level=16`; validate with uniform-state test B7 before DPF runs |
| R2 | **Sibling deref check missed**: `remove_child_blocks` called on one of 4 siblings while others remain, leaving the parent in a half-restricted state — coarse block has data from only 1/4 of its area | Medium (multi-block deref requires atomic check across all 4 children) | High | In `auto_regrid` deref pass, before calling `remove_child_blocks`, check that ALL `ratio^2` siblings have flag `-1`. Skip if any sibling is `0` or `+1`. Add test B6 that verifies partial-sibling deref is rejected. |
| R3 | **Regrid during RK substage invalidates ghost data**: if `auto_regrid` is called mid-RK3 step (impossible in current design but possible if step counter logic is wrong), new blocks get stale prolongated ghosts from pre-step coarse data | Low (prevented by calling at step boundary before RHS) | Medium | `amr_step` has a single entry point; `auto_regrid` is first operation before ghost exchange. Add assertion: `step_number % regrid_interval == 0` is the only condition. Add test B9 to verify regrid fires at correct steps. After regrid, always call `fill_all_ghosts()` before first RHS evaluation. |

---

## 8. File Changes Summary

| File | Action | New LOC |
|------|--------|---------|
| `src/dpf/metal/mlx_amr.py` | ADD: 6 new functions + 2 private helpers | +200 |
| `src/dpf/metal/mlx_amr.py` | EDIT: `amr_step()` — 3-line regrid hook at top | +3 |
| `src/dpf/metal/mlx_amr.py` | ADD: `_lohner_indicator_block`, `_current_density_sensor_block` (private 2D versions) | +60 |
| `src/dpf/config.py` | No changes — `AMRConfig` already added in Phase A | 0 |
| `tests/test_mlx_amr.py` | ADD: 10 new tests (B1-B10) | +200 |

**Total: ~350 new production LOC, ~200 new test LOC.**

---

## 9. Implementation Order

Execute sequentially. Each step must pass before proceeding.

1. Add private helpers `_lohner_indicator_block`, `_current_density_sensor_block` to `mlx_amr.py` (copy/adapt from prototype). Run `ruff check`.
2. Implement `evaluate_refinement_sensors`. Write test B1.
3. Implement `flag_blocks_for_refinement`. Write test B2.
4. Implement `enforce_proper_nesting`. Write test B3.
5. Implement `create_child_blocks`. Write tests B4, B5.
6. Implement `remove_child_blocks`. Write test B6.
7. Implement `auto_regrid`. Write tests B7, B8.
8. Wire `auto_regrid` into `amr_step()`. Write tests B9, B10.
9. Run full suite: `pytest tests/ -x -q -m "not slow"`. Zero new failures required.
10. Smoke-test on PF-1000 preset: `python3 -c "from dpf.metal.mlx_amr import auto_regrid; print('ok')"`. Verify no import errors.
