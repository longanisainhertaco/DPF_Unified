# AMR Phase A-slim: Implementation Specification

**Date**: 2026-03-26 | **Budget**: ~400 LOC prod + ~300 LOC tests | **Target**: <500 lines
**Prereqs**: `AMR_DESIGN_SCAFFOLD.md`, `amr_concerns_analysis.md`, `AMR_PROTOTYPE_CODE.md`

---

## 1. File Manifest

| File | Action | LOC |
|------|--------|-----|
| `src/dpf/metal/mlx_amr.py` | CREATE | ~400 |
| `src/dpf/config.py` | EDIT: add AMRConfig after AblationConfig (line 515) | +20 |
| `src/dpf/metal/mlx_solver.py` | EDIT: AMR dispatch in step() | +30 |
| `tests/test_mlx_amr.py` | CREATE | ~300 |
| `tests/conftest.py` | EDIT: add CylindricalGrid fixture (after line 139) | +8 |

---

## 2. Config: `src/dpf/config.py`

Insert after `AblationConfig` (line 515), before `KineticConfig`:

```python
class AMRConfig(BaseModel):
    """Block-structured AMR configuration."""
    enabled: bool = Field(False, description="Enable 2-level block AMR")
    max_levels: int = Field(2, ge=2, le=4)
    refinement_ratio: int = Field(2, ge=2, le=4)
    block_nr: int = Field(16, ge=8, le=64, description="Radial cells per block")
    block_nz: int = Field(32, ge=8, le=128, description="Axial cells per block")
    max_blocks_per_level: int = Field(16, ge=4, le=64)
    regrid_interval: int = Field(20, ge=1, le=200)
    j_threshold_refine: float = Field(0.3, gt=0, le=1.0)
    j_threshold_derefine: float = Field(0.05, ge=0, le=1.0)
    lohner_threshold_refine: float = Field(0.2, gt=0, le=1.0)
    lohner_threshold_derefine: float = Field(0.03, ge=0, le=1.0)
    buffer_width: int = Field(1, ge=0, le=3)
    use_refluxing: bool = Field(True, description="Berger-Colella flux correction at CF boundaries")
    refined_blocks: list[list[int]] | None = Field(None, description="Manual [[ir,iz],...] or None=auto")
```

Add to `SimulationConfig` (after line 602, with other sub-configs):
```python
    amr: AMRConfig = Field(default_factory=AMRConfig)
```

---

## 3. New File: `src/dpf/metal/mlx_amr.py`

### Module Header

```python
"""Block-structured AMR for MLX MHD solver (Phase A-slim).
2-level, global timestep, ghost exchange, prolongation/restriction, minimal refluxing.
Refs: Berger & Colella JCP 82:64 (1989), Stone et al. ApJS 249:4 (2020)."""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any
import numpy as np
logger = logging.getLogger(__name__)
try:
    import mlx.core as mx
except ImportError:
    mx = None
IDN, IMR, IMZ, IMT, IEN, ISR, IBR, IBZ, IBT, IEE = range(10)
NVAR = 10
_SIGN_FLIP_VARS = (IMR, IMT, IBR, IBT)  # sign-flipped at axis reflection
```

### 3.1 Data Structures (~80 LOC)

**`AMRBlock`** (10 LOC) -- fields: `level: int`, `index: tuple[int,int]`, `U: Any` (mx.array NVAR,nr,nz), `r_min: float`, `z_min: float`, `active: bool = True`.

**`AMRLevel`** (30 LOC) -- fields: `level: int`, `blocks: dict[tuple[int,int], AMRBlock]`, `dr: float`, `dz: float`. Methods: `active_blocks() -> list[AMRBlock]` (sorted by index), `as_batch() -> mx.array` (stack active U's), `scatter_batch(U_batch)` (distribute back).

**`AMRHierarchy`** (40 LOC) -- fields: `levels: list[AMRLevel]`, `block_nr: int`, `block_nz: int`, `ratio: int`. Methods: `n_levels` property, `total_cells() -> int`, `block_topology(level_idx) -> dict[idx, {N/S/E/W: idx|None}]` (neighbor map; N=+z, S=-z, E=+r, W=-r; None=physical boundary).

### 3.2 Domain Decomposition (~60 LOC)

**`decompose_domain(nr, nz, dr, dz, r_inner, block_nr, block_nz) -> AMRLevel`** (25 LOC)
Split grid into `ceil(nr/block_nr) x ceil(nz/block_nz)` blocks with zero-initialized U. Compute `r_min = r_inner + ir*block_nr*dr`, `z_min = iz*block_nz*dz` per block.

**`populate_blocks_from_state(level, U_global, block_nr, block_nz)`** (15 LOC)
Slice `U_global[:, ir*bnr:(ir+1)*bnr, iz*bnz:(iz+1)*bnz]` into each block.

**`assemble_global_state(level, nr, nz, block_nr, block_nz) -> mx.array`** (15 LOC)
Inverse: paste each block.U back into a `(NVAR, nr, nz)` array.

### 3.3 Ghost Exchange (~70 LOC)

**`ghost_exchange_same_level(level, ng, block_nr, block_nz, r_inner) -> dict[idx, mx.array]`**

For each active block, create padded array `(NVAR, nr+2*ng, nz+2*ng)`:
- Interior: `U_pad[:, ng:-ng, ng:-ng] = block.U`
- Each face: if neighbor exists, copy ng-wide slab from neighbor interior. Else apply physical BC:
  - **W (r=r_inner, axis)**: reflect with sign flip on `_SIGN_FLIP_VARS`
  - **E/S/N (outer/bottom/top)**: zero-gradient (copy last interior slab)
- Matches pattern from `MLXMHDSolver._pad_electrode_ghost()` (mlx_solver.py:266).
- Reuses: pattern only (no direct function call).

### 3.4 Prolongation (~50 LOC)

**`prolongate_to_fine(coarse_block, fine_level, ratio, block_nr, block_nz) -> list[AMRBlock]`** (20 LOC)
Each coarse block spawns `ratio^2` fine blocks. Extract coarse quadrant `(NVAR, bnr//ratio, bnz//ratio)`, call `_prolongate_vanleer()`, create child AMRBlock with correct r_min/z_min.

**`_prolongate_vanleer(U_coarse, ratio=2) -> np.ndarray`** (30 LOC)
Piecewise-linear with van Leer limiter. Algorithm from `AMR_PROTOTYPE_CODE.md:215-238`:
1. `np.repeat(U_coarse, ratio, axis=1/2)` for piecewise-constant base
2. For each interior cell: compute limited slopes `dr = vanleer(backward, forward)` in r and z
3. Sub-cell correction: `U_fine[v, i*R+di, j*R+dj] += dr*xi_r + dz*xi_z` where `xi = (di+0.5)/R - 0.5`

`_vanleer(a, b) -> float`: `2*a*b/(a+b) if a*b > 0 else 0.0`

### 3.5 Restriction (~40 LOC)

**`restrict_to_coarse(fine_blocks, coarse_block, fine_level, ratio, block_nr, block_nz)`**
Volume-weighted average with cylindrical `V = 0.5*(r_hi^2 - r_lo^2)*dz`. For each coarse cell, sum `U_fine * vol_fine` over the `ratio^2` covering fine cells, divide by total volume. Algorithm from `AMR_PROTOTYPE_CODE.md:241-265`.

### 3.6 Minimal Refluxing (~60 LOC)

Prevents 0.3-3% mass drift identified in `amr_concerns_analysis.md`.

**`FluxRegister`** (20 LOC) -- dataclass with `coarse_FA: dict[int, ndarray]`, `fine_FA: dict[int, ndarray]`. Methods: `reset()`, `accumulate_coarse(fid, flux, area, dt)`, `accumulate_fine(fid, flux, area, dt)`.

**`identify_cf_faces(hierarchy, coarse_li=0) -> list[dict]`** (20 LOC)
Find coarse cell faces adjacent to fine blocks. Each dict: `face_id, coarse_block_idx, coarse_cell, face_dir(r|z), face_side(lo|hi), fine_faces, coarse_area, coarse_volume`. Areas: `A_r = r*dz`, `A_z = 0.5*(r_hi^2 - r_lo^2)`, `V = 0.5*(r_hi^2 - r_lo^2)*dz`.

**`apply_reflux_correction(register, cf_faces, coarse_level) -> float`** (20 LOC)
For each face: `delta = fine_FA - coarse_FA`, `sign = +1 if hi else -1`, `U_c[:, ir, iz] += sign * delta / V_c`. Returns total |correction| for monitoring. Phase A uses **re-evaluation** at CF faces (re-compute Riemann flux from reconstructed states) rather than capturing fluxes from `mhd_rhs()` -- avoids modifying `mlx_riemann.py`.

### 3.7 Orchestrator (~70 LOC)

**`build_amr_hierarchy(nr, nz, dr, dz, r_inner, block_nr, block_nz, ratio, refined_blocks=None) -> AMRHierarchy`** (30 LOC)
Create 2-level hierarchy. Level 0 via `decompose_domain()`. Level 1 at `dr/ratio, dz/ratio`. If `refined_blocks` provided, prolongate specified coarse blocks to create fine children. Else level 1 starts empty.

**`amr_step(hierarchy, dt, gamma, method, riemann, ng, current, r_inner, step_number, rhs_fn, use_refluxing=True) -> (AMRHierarchy, float)`** (40 LOC)

Pipeline per step:
1. Ghost exchange on all levels (same-level + prolongation from coarse for level 1)
2. Global CFL dt = min across all blocks/levels
3. Advance each level: per-block RHS on padded state, SSP-RK3, strip ghosts. If refluxing: accumulate face fluxes in FluxRegister
4. Restrict fine -> coarse (volume-weighted)
5. Apply reflux correction at CF faces

Block processing is sequential (not vmap) in Phase A. vmap batching deferred to Phase D.

---

## 4. Solver Integration: `src/dpf/metal/mlx_solver.py`

**Constructor** (line ~131): add `amr_config: AMRConfig | None = None` param. Store as `self._amr_config`. Initialize `self._amr_hierarchy = None`, `self._step_count = 0`.

**step()** (line ~585): insert at top, before existing pipeline:
```python
if self._amr_config is not None and self._amr_config.enabled:
    return self._step_amr(state, dt, current, voltage, **kwargs)
```

**`_step_amr()`** (~25 LOC): On first call, build hierarchy from global state via `build_amr_hierarchy()` + `populate_blocks_from_state()`. Call `amr_step()`. Reconstruct global state via `assemble_global_state()`. Return state dict via `_state_mgr.to_state_dict()`.

---

## 5. Conftest Addition: `tests/conftest.py`

Insert after line 139 (`dx` fixture):
```python
@pytest.fixture
def cylindrical_grid():
    """Small CylindricalGrid for AMR unit tests."""
    mlx = pytest.importorskip("mlx.core")
    from dpf.metal.mlx_grid import CylindricalGrid
    return CylindricalGrid(nr=32, nz=64, dr=1e-3, dz=1e-3, r_inner=0.01)
```

---

## 6. Tests: `tests/test_mlx_amr.py`

All tests gate on `pytest.importorskip("mlx.core")`. Constants: `NVAR=10, BNR=16, BNZ=32`.

| # | Function | Tests | Key assertion |
|---|----------|-------|---------------|
| 1 | `test_decompose_block_count` | `decompose_domain(32,64,1e-3,1e-3,0.01,16,32)` | 2x2 = 4 blocks |
| 2 | `test_decompose_coordinates` | Block physical positions | `b(1,0).r_min == 0.01 + 16*1e-3` |
| 3 | `test_populate_assemble_roundtrip` | populate -> assemble == identity | `max|diff| < 1e-6` |
| 4 | `test_ghost_interior_copy` | Neighbor data appears in ghost cells | `ghost_slab == neighbor_interior_slab` |
| 5 | `test_ghost_axis_reflection` | Sign flip at r=0 for B_r, B_theta | `ghost[IBR] == -interior[IBR]` |
| 6 | `test_ghost_outflow_zerograd` | Edge blocks get zero-gradient ghosts | `ghost == last_interior` |
| 7 | `test_prolong_constant` | Uniform field preserved | `all(fine == 1.0)` |
| 8 | `test_prolong_linear` | Linear profile accurate | `L_inf < O(dr^2)` |
| 9 | `test_restrict_recovers_coarse` | `restrict(prolongate(U)) ~ U` for smooth data | `L_inf < 1e-3` |
| 10 | `test_restrict_conserves_mass` | `sum(rho*V)` preserved | `|delta| < 1e-6 relative` |
| 11 | `test_amr_step_uniform_preserved` | Uniform state unchanged after 1 step | `max|dU| < 1e-6` |
| 12 | `test_amr_step_mass_conservation` | Mass conserved with refluxing (Sod across blocks) | `|dm/m| < 1e-5` |
| 13 | `test_flux_register` | FluxRegister accumulates correctly | `delta == fine_sum - coarse` |
| 14 | `test_reflux_sign` | Correction sign correct for hi/lo | Coarse cell gains mass when fine > coarse |
| 15 | `test_build_hierarchy_manual` | `refined_blocks=[(0,1)]` | Level 1 has 4 children |
| 16 | `test_hierarchy_dr_dz` | `level1.dr == dr_base / ratio` | Exact equality |
| 17 | `test_amr_disabled_default` | `AMRConfig(enabled=False)` -> no hierarchy | `solver._amr_hierarchy is None` |
| 18 | `test_existing_solver_unaffected` | MLXMHDSolver without amr_config | Results match non-AMR baseline |

---

## 7. Implementation Order

1. **Config** -> run existing tests (AMR off by default, zero risk)
2. **Data structures** (AMRBlock/Level/Hierarchy, decompose, populate, assemble) + tests 1-3
3. **Ghost exchange** + tests 4-6
4. **Prolongation/restriction** + tests 7-10
5. **Refluxing** (FluxRegister, identify_cf_faces, apply_correction) + tests 13-14
6. **Orchestrator** (build_hierarchy, amr_step) + tests 11-12, 15-16
7. **Solver wiring** (mlx_solver.py changes) + tests 17-18
8. **Full suite**: `pytest tests/ -x -q` -- all ~4900 existing tests pass

---

## 8. Frozen Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Block size default | 16x32 | 6.25% min refined area (concerns rec. 3) |
| Ghost width | 3 | Matches `MLXMHDSolver._GHOST_NG` (WENO5-Z) |
| Refluxing | Phase A (not deferred) | 0.3-3% mass drift without it |
| Subcycling | Deferred to Phase C | 2x CFL penalty acceptable |
| div(B) at CF boundary | Dedner GLM | CT across levels unsolved for GPU |
| Block processing | Sequential (not vmap) | Phase A simplicity; vmap in Phase D |
| Flux capture | Re-evaluate at CF faces | Avoids modifying `mlx_riemann.py` signature |

---

## 9. Acceptance Criteria

| # | Criterion | Threshold |
|---|-----------|-----------|
| 1 | Existing test suite | 0 new failures |
| 2 | Populate/assemble roundtrip | max error < 1e-6 |
| 3 | Mass conservation (Sod, with refluxing) | |dm/m| < 1e-5 per step |
| 4 | Uniform state preservation (10 steps) | max |dU| < 1e-5 |
| 5 | Restriction mass conservation | < 1e-6 relative |
| 6 | Prolongation of constants | < 1e-7 |
| 7 | Axis BC sign flip | Exact |
| 8 | Fine level resolution | dr_fine == dr_coarse / ratio (exact) |
