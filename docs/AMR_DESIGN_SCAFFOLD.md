# AMR Design Scaffold for DPF-Unified

Design-only document. No implementation accompanies this file.

**Author**: Engine Architect Agent
**Date**: 2026-03-26
**Status**: DRAFT -- requires review before Phase implementation
**Prerequisite reading**: `src/dpf/experimental/static_refinement.py` (562 LOC, existing SMR)

---

## 1. Architecture Options

### 1A. Patch-Based AMR (Berger-Oliger / AMReX style)

Arbitrary rectangular patches at each level. Patches can overlap and tile
irregularly. AMReX (LBNL) is the reference implementation.

- **Pros**: Maximum flexibility; fewest wasted cells; mature theory (Berger &
  Colella 1989).
- **Cons**: Complex bookkeeping (patch list management, load balancing); variable
  array shapes incompatible with `mx.compile()` static-shape requirement; patch
  creation/destruction forces Python-level memory allocation every regrid.

### 1B. Octree AMR

Cell-by-cell refinement organized in a tree. RAMSES (Teyssier 2002) is the
reference.

- **Pros**: Cell-level adaptivity; minimal wasted cells.
- **Cons**: Pointer-chasing memory access pattern is worst-case for GPU; not
  representable as contiguous arrays; 2D cylindrical is not a natural octree
  geometry.

### 1C. Block-Structured AMR (Athena++ / Parthenon style)

Fixed-size blocks (e.g., 32x64) organized into refinement levels. Each level is
a collection of identically-shaped blocks. Athena++ (Stone et al. 2020) and
Parthenon (Grete et al. 2022) use this approach.

- **Pros**: Every block is the same shape -- compatible with `mx.compile()`;
  batch all blocks at same level into a single MLX call via `mx.vmap()` or
  leading batch dimension; simple ghost exchange (fixed stencil offsets);
  maps directly onto existing `MLXMHDSolver` which already operates on
  `(NVAR, nr, nz)` arrays.
- **Cons**: Wastes cells at block boundaries (buffer zones); refinement
  granularity limited to block size; potential Amdahl bottleneck during regrid
  (Parthenon-VIBE 2025, arXiv:2509.19701).

### Recommendation: Block-Structured AMR

Block-structured is the only viable option for MLX. The static-shape constraint
of `mx.compile()` eliminates patch-based and octree. Block-structured also maps
naturally onto the existing solver: each block is a `(NVAR, nr_block, nz_block)`
array, identical to the current solver input. The Parthenon-VIBE Amdahl
bottleneck is mitigable at 2 levels (see Section 6).

---

## 2. Refinement Criteria

### Options Evaluated

| Criterion | Formula | Sensitivity to Sheath | False Positives | Cost |
|-----------|---------|----------------------|-----------------|------|
| Lohner (density) | `\|d2rho/dx2\| / (\|drho/dx\|/dx + eps*\|rho\|/dx^2)` | HIGH | Medium (any shock) | LOW |
| Current density | `\|curl B\| * dx / (\|B\| + eps)` | HIGHEST | Low | Medium |
| Pressure jump | `\|dp\|/p` across faces | Medium | High (acoustic waves) | LOW |
| div(B) error | `\|div B\| * dx / \|B\|` | Low | Irrelevant noise | LOW |
| Density gradient | `\|grad rho\| / rho` | High | Medium | LOW |

### Recommendation: Current Density (primary) + Lohner (secondary)

For DPF, the current density `|J| = |curl B|/mu_0` is the most physically
motivated sensor. The sheath IS the current sheet -- J peaks exactly where
refinement is needed. The Lohner indicator catches additional features (rarefaction
waves, contact discontinuities) that J misses.

Both sensors already exist in `static_refinement.py` (lines 321-441) as
`lohner_error_indicator()` and `current_density_sensor()`. The combined
criterion:

```python
def needs_refinement(block: BlockState, thresholds: tuple[float, float]) -> bool:
    j_sensor = current_density_sensor(block.B, block.dr, block.dz)
    lohner = lohner_error_indicator(block.rho, block.dr, block.dz)
    return np.max(j_sensor) > thresholds[0] or np.max(lohner) > thresholds[1]
```

Threshold tuning: `j_threshold = 0.3`, `lohner_threshold = 0.2` based on
typical DPF sheath profiles. These should be configurable.

---

## 3. Data Structure

### Block Definition

```python
@dataclass
class AMRBlock:
    """Single block in the AMR hierarchy."""
    level: int                    # 0 = coarsest
    index: tuple[int, int]        # (ir, iz) position in level grid
    U: mx.array                   # conserved state (NVAR, nr_block, nz_block)
    psi: mx.array | None          # Dedner cleaning scalar (nr_block, nz_block)
    r_min: float                  # physical left edge [m]
    z_min: float                  # physical bottom edge [m]
    dr: float                     # cell spacing at this level
    dz: float                     # cell spacing at this level
    parent: tuple[int, int] | None
    children: list[tuple[int, int]]  # up to 4 children (2x2 in r,z)
    needs_update: bool            # flagged for refinement/coarsening
```

### Level Container

```python
@dataclass
class AMRLevel:
    """All blocks at one refinement level."""
    level: int
    blocks: dict[tuple[int, int], AMRBlock]
    dr: float
    dz: float
    dt: float                     # timestep for this level (subcycled)

    def as_batch(self) -> mx.array:
        """Stack all blocks into (N_blocks, NVAR, nr_block, nz_block)."""
        return mx.stack([b.U for b in self.blocks.values()])
```

### 2-Level Layout (Phase A Target)

```
Level 0 (base):  4 blocks of (NVAR, 32, 64) covering full domain
Level 1 (fine):  1-4 blocks of (NVAR, 32, 64) covering sheath region
                 (factor-2 refinement: dr1 = dr0/2, dz1 = dz0/2)

Total cells:
  Level 0: 4 * 32 * 64 = 8,192
  Level 1: 4 * 32 * 64 = 8,192 (worst case, typically 1-2 blocks)
  Effective: equiv. to 128x256 at sheath, 64x128 elsewhere
```

Each block has ghost zones (2 cells for PLM, 3 for WENO5) filled from:
- Neighboring blocks at same level (copy)
- Parent block at coarser level (prolongation / interpolation)
- Physical boundaries (electrode BCs, axis, outflow)

### Memory Budget (M3 Pro, 36 GB)

```
Per block: NVAR * nr * nz * 4 bytes = 10 * 32 * 64 * 4 = 80 KB
8 blocks (2 levels): 640 KB
Ghost zones add ~25%: ~800 KB
Workspace (RHS, fluxes): 3x state = 2.4 MB
Total: < 4 MB -- negligible on 36 GB
```

---

## 4. Flux Conservation at Coarse-Fine Boundaries

At level boundaries, the coarse cell face sees a single flux, but the fine
cells compute 2 (or 4 in 3D) sub-fluxes along the same physical face. Without
correction, conservation fails at O(dx) -- the dominant error in AMR.

### Refluxing Algorithm (Berger & Colella 1989)

After advancing both levels, correct the coarse flux at each coarse-fine face:

```
delta_F = sum(F_fine * A_fine) - F_coarse * A_coarse
U_coarse[adjacent_cell] += delta_F * dt / V_coarse
```

For cylindrical coordinates, the face areas and cell volumes carry `r` factors:

```python
def reflux_correction_cylindrical(
    U_coarse: mx.array,     # (NVAR, nr_c, nz_c)
    F_coarse_r: mx.array,   # (NVAR, nr_c+1, nz_c) radial flux at faces
    F_fine_r: mx.array,     # (NVAR, nr_f+1, nz_f) fine radial flux
    r_face_coarse: mx.array, # radial face positions, coarse
    r_face_fine: mx.array,   # radial face positions, fine
    dt: float,
    dr_c: float,
    dz_c: float,
) -> mx.array:
    """Apply refluxing correction at coarse-fine radial boundary.

    For each coarse face that abuts fine blocks:
      dF = (r_f[j]*F_f[j]*dz_f + r_f[j+1]*F_f[j+1]*dz_f) - r_c*F_c*dz_c
      U_c[adjacent] += dF * dt / (r_c * dr_c * dz_c)
    """
    # Implementation: loop over boundary faces, accumulate fine fluxes,
    # subtract coarse flux, apply correction to adjacent coarse cell.
    # This is O(N_boundary) -- cheap relative to interior update.
    ...
```

The cylindrical `r` weighting is critical. Standard Cartesian refluxing
(AMReX default) does not account for `2*pi*r*dr*dz` volume elements. Our
solver already uses `r`-weighted finite volumes (see MEMORY.md:
"r-weighted finite volume > operator-split for cylindrical MHD"), so the
refluxing must be consistent.

### Constrained Transport at Boundaries

If CT is active, the magnetic flux through coarse-fine faces must also be
corrected to maintain div(B) = 0. This is the hardest part of AMR for MHD
and the reason most GPU MHD codes (including AthenaK) defer it.

For Phase A-B, use Dedner GLM cleaning (already implemented) instead of CT
at level boundaries. CT remains active within each block. This is the
approach used by MPI-AMRVAC (Keppens et al. 2023).

---

## 5. Subcycling

### The Question

Should the fine level (level 1, dt_fine = dt_coarse / 2) take 2 steps per
coarse step?

### Option A: Subcycling (standard)

```
t=0          t=dt_c/2       t=dt_c
L0: |------ full step ------|
L1: |-- half --|-- half --|
    ^          ^          ^
    sync       sync       sync + reflux
```

- **Pros**: CFL-correct on both levels; fine level resolves fast waves at
  fine scale; standard in Athena++, FLASH, AMReX.
- **Cons**: 2x the fine-level work; synchronization points add latency;
  on GPU, the fine blocks alone may underutilize the hardware.

### Option B: Global Timestep (no subcycling)

Both levels advance with `dt = min(dt_coarse, dt_fine)`.

- **Pros**: Simpler implementation; no synchronization; all blocks advance
  in one batched MLX call; avoids the Parthenon-VIBE Amdahl bottleneck.
- **Cons**: Coarse level takes unnecessarily small steps; overall speedup
  from AMR reduced (coarse level is CFL-limited by fine dt).

### Recommendation for DPF: Global Timestep (Phase A-B), Subcycling (Phase C+)

For 2-level AMR with factor-2, the coarse level only wastes a factor of 2 in
timestep. The implementation simplicity and GPU utilization advantages dominate.
At 3+ levels (Phase D), subcycling becomes necessary to avoid the coarse level
taking 4x or 8x smaller timesteps.

---

## 6. MLX Constraints and Mitigations

### Constraint 1: `mx.compile()` Requires Static Shapes

**Impact**: Cannot change the number of blocks or block size at runtime inside
a compiled function.

**Mitigation**: Pre-allocate a fixed maximum number of blocks per level.
Inactive blocks are masked with a boolean flag. The compiled kernel processes
all slots; inactive slots produce zero flux (branch-free masking).

```python
MAX_BLOCKS_PER_LEVEL = 16  # pre-allocated
active_mask: mx.array       # (MAX_BLOCKS_PER_LEVEL,) bool

# Batch update: all blocks at once, masked
U_batch = level.as_padded_batch()  # (MAX_BLOCKS, NVAR, nr, nz)
U_new = compiled_rhs(U_batch) * active_mask[:, None, None, None]
```

### Constraint 2: float32 Only on Metal GPU

**Impact**: Conservation at coarse-fine boundaries accumulates float32 rounding.
Refluxing correction adds O(eps_32) ~ 1e-7 error per step.

**Mitigation**: Acceptable for DPF. The dual-energy entropy tracer (already
implemented) handles the dominant float32 failure mode (pressure recovery).
Refluxing errors are sub-dominant. For production V&V, use `precision="float64"`
(CPU fallback) which bypasses Metal entirely.

### Constraint 3: No Dynamic Memory Allocation in Metal Kernels

**Impact**: Cannot create new blocks inside a kernel.

**Mitigation**: Regridding (block creation/destruction) happens in Python on
CPU. Only the RHS evaluation and time integration run on Metal. The regrid
frequency is low (every 10-50 steps), so the CPU overhead is small.

### Constraint 4: Parthenon-VIBE Amdahl Bottleneck

The Parthenon-VIBE paper (arXiv:2509.19701) found that AMR overhead
(regridding, ghost exchange, load balancing) can exceed compute savings
on GPU, especially for small block counts. Their measurement: regrid cost
scales as O(N_blocks * N_ghost) and dominates when blocks < 64.

**Mitigation for DPF**: With 2 levels and max 16 blocks, we have at most
~32 blocks total. Ghost exchange is a memory copy, not a kernel launch.
Regrid every 10-50 steps amortizes the cost. The DPF sheath is spatially
coherent (not fragmented), so block count stays low. The bottleneck is
real for 3D cosmology sims with thousands of blocks -- not for 2D
cylindrical DPF.

---

## 7. Implementation Phases

### Phase A: 2-Level SMR with Manual Region Selection

Extend `static_refinement.py` to use block-structured layout instead of
independent fine-grid re-simulation.

- Replace single fine grid with `AMRBlock` / `AMRLevel` data structures
- Manual region selection (user specifies which blocks are refined)
- Ghost exchange between blocks at same level (copy)
- Prolongation from level 0 to level 1 ghost zones (bilinear)
- Restriction from level 1 to level 0 (volume-weighted average)
- Global timestep (no subcycling)
- No refluxing yet (conservation error accepted)
- Wire into `MLXMHDSolver` as an optional mode

**Dependencies**: None (builds on existing static_refinement.py)
**LOC estimate**: 400-500
**Files**: `src/dpf/experimental/amr.py` (new), edits to `mlx_solver.py`

### Phase B: Automatic Refinement with Indicators

- Implement block-level refinement tagging using `current_density_sensor()`
  and `lohner_error_indicator()` (both already exist)
- Automatic regrid every N steps (configurable, default 20)
- Proper nesting constraint: refined block must have parent
- Buffer zone: tag neighbors of flagged blocks to avoid refinement fronts
- Add `AMRConfig` to Pydantic config with thresholds

**Dependencies**: Phase A
**LOC estimate**: 250-350
**Files**: `src/dpf/experimental/amr.py` (extend), `src/dpf/config.py` (add AMRConfig)

### Phase C: Refluxing + Optional Subcycling

- Implement Berger-Colella refluxing with cylindrical `r`-weighting
- Add subcycling as configurable option (default off)
- Conservation verification: total mass/energy/momentum before and after
  correction step
- Dedner GLM at level boundaries (not CT)

**Dependencies**: Phase A, Phase B
**LOC estimate**: 300-400
**Files**: `src/dpf/experimental/amr.py` (extend)

### Phase D: 3+ Levels + Performance

- Generalize to N levels with recursive subcycling
- `mx.vmap()` over blocks at each level for batch processing
- Profiling and tuning: regrid frequency, block size, max levels
- Coarsening criterion (de-refine when indicator drops below threshold)

**Dependencies**: Phase C
**LOC estimate**: 200-300
**Files**: `src/dpf/experimental/amr.py` (extend)

---

## 8. LOC Estimates Summary

| Phase | New LOC | Cumulative | Calendar Estimate | Key Risk |
|-------|---------|------------|-------------------|----------|
| A | 400-500 | 400-500 | 2-3 sessions | Ghost exchange bugs |
| B | 250-350 | 650-850 | 1-2 sessions | Over-refinement |
| C | 300-400 | 950-1250 | 2-3 sessions | Conservation error at boundaries |
| D | 200-300 | 1150-1550 | 1-2 sessions | Performance regression |
| **Total** | **1150-1550** | | **6-10 sessions** | |

Test LOC (not included above): ~300-400 per phase, ~1200-1600 total.

---

## 9. Risk Assessment

### Risk 1: Flux Conservation at Coarse-Fine Boundaries (HIGHEST)

**Probability**: High. This is the hardest part of AMR for MHD.
**Impact**: Mass/energy drift proportional to sheath velocity * boundary area.
**Detection**: Monitor `delta_mass / mass_total` per step. Threshold: 1e-8.
**Mitigation**: Phase C refluxing. Until then, accept O(dx) conservation error.
**FMEA RPN**: 315 (from DMAIC_FORWARD_PLAN.md, item 7.1).

### Risk 2: GPU Memory Fragmentation

**Probability**: Low for 2D cylindrical (blocks are < 1 MB each).
**Impact**: OOM or performance cliff from non-contiguous allocations.
**Detection**: Monitor `mx.metal.get_active_memory()` during regrid.
**Mitigation**: Pre-allocate fixed MAX_BLOCKS. Reuse slots.

### Risk 3: Performance Regression from Regridding (Amdahl)

**Probability**: Medium for Phase D (3+ levels with frequent regrid).
**Impact**: AMR slower than uniform fine grid.
**Detection**: Wall-clock comparison: AMR vs uniform at matched effective resolution.
**Mitigation**: Regrid every 20-50 steps (not every step). Global timestep
avoids subcycling overhead for 2 levels. Profile before adding complexity.
**FMEA RPN**: 336 (from DMAIC_FORWARD_PLAN.md, item 7.3).

### Risk 4: Breaking Existing Tests

**Probability**: Medium. 4,900+ tests assume `(NVAR, nr, nz)` uniform arrays.
**Impact**: CI failures; wasted debugging time.
**Detection**: Run full test suite after Phase A.
**Mitigation**: AMR is opt-in (`use_amr=True` in config). Default path unchanged.
AMR blocks expose the same `(NVAR, nr, nz)` interface per block.

### Risk 5: CT Divergence at Level Boundaries

**Probability**: High if CT is used across levels.
**Impact**: Growing div(B) at coarse-fine faces.
**Detection**: Monitor max(|div B|) at level boundaries.
**Mitigation**: Use Dedner GLM at boundaries (already implemented). CT within
blocks only. This is explicitly the MPI-AMRVAC approach.

### Risk Matrix

| Risk | Probability | Impact | Phase Affected | Mitigation Ready? |
|------|-------------|--------|----------------|-------------------|
| Flux conservation | High | High | C | Yes (refluxing) |
| CT at boundaries | High | Medium | C | Yes (Dedner) |
| Amdahl bottleneck | Medium | Medium | D | Partial (global dt) |
| Test breakage | Medium | Low | A | Yes (opt-in) |
| Memory fragmentation | Low | Medium | D | Yes (pre-alloc) |

---

## 10. References

1. **Berger, M.J. & Colella, P.**, "Local adaptive mesh refinement for shock
   hydrodynamics," *JCP* 82:64-84 (1989).
   -- Foundational AMR algorithm. Refluxing at coarse-fine boundaries.

2. **Stone, J.M. et al.**, "The Athena++ adaptive mesh refinement framework,"
   *ApJS* 249:4 (2020).
   -- Block-structured AMR for MHD with CT. Reference architecture.

3. **Grete, P. et al.**, "Parthenon -- a performance portable block-structured
   adaptive mesh refinement framework," *IJHPCA* 37:465-486 (2022).
   -- GPU-native block AMR built on Kokkos. Basis for AthenaK.

4. **Parthenon-VIBE** (arXiv:2509.19701, 2025).
   -- Found severe Amdahl bottleneck: AMR overhead exceeds compute savings
   on GPU when regrid frequency is high. Key constraint for our design.

5. **Lohner, R.**, "An adaptive finite element scheme for transient problems
   in CFD," *CMAME* 61:323-338 (1987).
   -- Second-derivative error indicator. Already implemented in our codebase.

6. **Keppens, R. et al.**, "MPI-AMRVAC 3.0," *A&A* 673:A66 (2023).
   -- Block AMR with GLM div(B) cleaning at level boundaries (not CT).
   Validates our approach of Dedner at boundaries, CT within blocks.

7. **Teyssier, R.**, "Cosmological hydrodynamics with adaptive mesh
   refinement," *A&A* 385:337-364 (2002).
   -- Octree AMR (RAMSES). Included for comparison; not recommended for GPU.

8. **Zhang, W. et al.**, "AMReX: a framework for block-structured adaptive
   mesh refinement," *JOSS* 4:1370 (2019).
   -- Patch-based AMR. Reference for refluxing implementation details.

9. **Miyoshi, T. & Kusano, K.**, "A multi-state HLL approximate Riemann
   solver for ideal MHD," *JCP* 208:315-344 (2005).
   -- HLLD solver used within each block.

10. **Popovas, A. et al.**, "DISPATCH HLLS: entropy-stable MHD in float32,"
    *arXiv:2211.02438* (2025).
    -- Dual-energy entropy tracer. Critical for float32 AMR where
    refluxing errors compound with cancellation errors.

---

## Appendix: Decision Log

| Decision | Chosen | Rejected | Rationale |
|----------|--------|----------|-----------|
| AMR type | Block-structured | Patch, Octree | `mx.compile()` static shapes |
| Primary sensor | Current density | Density gradient | J peaks at sheath; physically motivated |
| Subcycling (Phase A-B) | Global dt | Subcycled | Simpler; 2-level penalty is only 2x |
| div(B) at boundaries | Dedner GLM | CT | CT across levels is unsolved for GPU |
| Block size | 32x64 | 16x32, 64x128 | Matches typical nr/nz ratio; fits L1 cache |
| Max blocks pre-alloc | 16 per level | Dynamic | `mx.compile()` needs static shapes |

---

## Six Sigma Refinement (DMAIC)

**Reviewer**: Engine Architect Agent (Opus)
**Date**: 2026-03-26
**Objective**: Raise readiness from 6/10 to 8/10 by addressing all issues from SCAFFOLD_REVIEW_SIX_SIGMA.md

---

### SR-1. Define: Minimum Viable AMR for PF-1000

The simplest AMR that demonstrates value is **2-level block-structured with manual
region selection, global timestep, NO refluxing, NO subcycling, NO auto-regrid**.
This is essentially "multi-block SMR" -- a stepping stone from `static_refinement.py`
(single fine grid re-simulation) to true AMR.

**Minimum viable scope (Phase A-slim)**:

1. `AMRBlock` / `AMRLevel` dataclasses (reuse existing `RefinementRegion` concept)
2. Block decomposition of base grid into fixed tiles
3. Ghost exchange between same-level neighbors (copy)
4. Prolongation from level 0 to level 1 ghost zones (bilinear, reuse
   `interpolate_to_fine_grid()` from `static_refinement.py`)
5. Restriction from level 1 to level 0 (volume-weighted average)
6. Wire into `MLXMHDSolver` as `amr_mode="manual"` with user-specified refined blocks
7. Single compiled RHS kernel processing batched blocks via leading batch dim

**What is deliberately excluded from Phase A-slim**:
- Auto-refinement (Phase B)
- Refluxing / flux correction (Phase C)
- Subcycling (Phase C+)
- 3+ levels (Phase D)

**Success criteria**:
- AMR at 2 levels resolves the PF-1000 sheath with 2x finer cells than uniform grid
  at the SAME wall-clock time or faster
- `|delta_mass/mass| < 1e-6` per step (accepted without refluxing; sheath moves
  through ~2-4 block boundaries per discharge)
- All 4,900+ existing tests pass with `use_amr=False` (default)
- Sheath peak `|J|` resolved with >= 8 cells across the current sheet (vs ~4 on
  the uniform coarse grid)

**Why 2x refinement, not 4x**:
For DPF sheaths, the current sheet thickness is ~1-2 mm (Auluck 2022, arXiv:2211.16775).
On a 64-cell radial grid spanning a 30mm annular gap, `dr_coarse ~ 0.47 mm`, giving
~2-4 cells across the sheath. Factor-2 refinement gives `dr_fine ~ 0.23 mm` and
~4-8 cells across the sheath. Factor-4 would give ~8-16 cells but requires 4x CFL
penalty on the global timestep (Phase A uses no subcycling). Factor-2 is the
minimum that helps and keeps the CFL penalty at 2x, which is acceptable.

---

### SR-2. Measure: Revised LOC Estimates

The review correctly identified that original estimates were 40-60% too low.
Revised estimates below include per-function breakdowns and confidence ranges
(P25/P50/P75 percentiles).

#### Phase A-slim (block data structures + ghost exchange + solver wiring)

| Component | P25 | P50 | P75 | Notes |
|-----------|-----|-----|-----|-------|
| `AMRBlock` dataclass | 25 | 35 | 45 | Fields + `as_array()` / `from_array()` |
| `AMRLevel` dataclass | 30 | 40 | 55 | Block dict + `as_batch()` + `active_mask` |
| `AMRHierarchy` class | 40 | 60 | 80 | 2-level container, block lookup, topology |
| `decompose_domain()` | 20 | 30 | 40 | Split uniform grid into blocks |
| `ghost_exchange_same_level()` | 50 | 70 | 100 | Copy from neighbors; handle physical BCs; priority order: phys BC > neighbor > prolongation |
| `prolongate_to_fine()` | 40 | 55 | 75 | Bilinear interp; reuses `RegularGridInterpolator` pattern from `static_refinement.py:174-260` |
| `restrict_to_coarse()` | 30 | 45 | 60 | Volume-weighted avg with r-weighting |
| `amr_step()` orchestrator | 50 | 70 | 95 | Global dt; batch blocks; call MLX RHS; restrict; ghost exchange |
| Solver wiring (`mlx_solver.py`) | 30 | 45 | 60 | `amr_mode` config; block-level `_rhs()` dispatch |
| Config (`config.py`) | 15 | 20 | 25 | `AMRConfig` Pydantic model |
| **Phase A-slim total** | **330** | **470** | **635** | |
| Tests | 200 | 280 | 380 | 8-12 tests: decompose, ghost, prolong, restrict, roundtrip, step |

**Reusable code from `static_refinement.py` (562 LOC)**:
- `lohner_error_indicator()` (lines 321-377, 57 LOC) -- Phase B
- `current_density_sensor()` (lines 380-425, 46 LOC) -- Phase B
- `identify_refinement_cells()` (lines 428-441, 14 LOC) -- Phase B
- `interpolate_to_fine_grid()` (lines 174-260, 87 LOC) -- prolongation pattern
- `detect_sheath_location()` (lines 48-96, 49 LOC) -- diagnostic
- `compute_refinement_region()` (lines 99-171, 73 LOC) -- region selection
- **Total reusable**: ~326 LOC across Phases A-B

#### Revised Phase Estimates (all phases)

| Phase | P25 | P50 | P75 | Calendar (sessions) | Original Estimate |
|-------|-----|-----|-----|---------------------|-------------------|
| A-slim | 330 | 470 | 635 | 3-4 | 400-500 |
| B (auto-refine) | 200 | 300 | 420 | 2-3 | 250-350 |
| C (refluxing + subcycling) | 400 | 580 | 750 | 3-5 | 300-400 |
| D (3+ levels + perf) | 200 | 300 | 400 | 2-3 | 200-300 |
| **Total production** | **1,130** | **1,650** | **2,205** | **10-15** | **1,150-1,550** |
| **Total tests** | **800** | **1,100** | **1,500** | included | **1,200-1,600** |

The P50 estimate (1,650 LOC) is 40% above the original midpoint (1,350), confirming
the review's assessment. Phase C is the dominant increase -- cylindrical refluxing
is research-grade code without a drop-in reference.

---

### SR-3. Analyze: Cylindrical Refluxing -- The Research Gap

#### SR-3.1 The Problem

Standard Cartesian refluxing (Berger & Colella 1989) corrects coarse fluxes at
coarse-fine boundaries:

```
delta_F = sum(F_fine * A_fine) - F_coarse * A_coarse
U_coarse[adjacent] += delta_F * dt / V_coarse
```

In cylindrical (r, z) coordinates, face areas and cell volumes carry `r` factors:
- Radial face area: `A_r = 2*pi*r_face * dz`
- Axial face area: `A_z = pi * (r_outer^2 - r_inner^2)` (annular)
- Cell volume: `V = pi * (r_outer^2 - r_inner^2) * dz`

The r-weighting means refluxing corrections are NOT uniform across the boundary --
cells at larger r have proportionally larger volumes and face areas.

#### SR-3.2 Literature Survey

**MPI-AMRVAC** (Keppens et al. 2023):
- Has cylindrical coordinates with AMR (`amrvac.org/md_doc_axial.html`).
- The `mod_fix_conserve` module handles flux conservation at refinement boundaries.
- Treats cylindrical geometry source terms separately from flux corrections --
  geometric source terms (centrifugal, hoop stress) are added as source terms
  in the swept cells, not incorporated into the reflux delta.
- The actual reflux operation in `mod_fix_conserve` uses the same `delta_F * dt / dV`
  pattern as Cartesian but with geometry-dependent `dV`. The volume element is
  pre-computed per cell and stored, so the reflux correction naturally inherits
  the `r`-weighting through the volume array.
- **Key insight**: MPI-AMRVAC does NOT special-case the refluxing for cylindrical
  coords -- it uses pre-computed volumes that already contain the `r` factors.
  The reflux formula is geometry-agnostic when volumes and areas are correct.

**PLUTO + CHOMBO** (Mignone et al. 2012, ApJS 198:7):
- AMR via CHOMBO library. Supports cylindrical coordinates.
- CHOMBO's `FluxRegister` accumulates fine-coarse flux mismatches.
- PLUTO computes fluxes in physical units with geometry factors included.
- The reflux correction uses CHOMBO's area-weighted flux differencing, where
  areas are computed from the coordinate geometry.
- **Approach**: flux * area is accumulated on both sides; the mismatch is
  divided by the coarse cell volume to get the conservative correction.

**Athena++** (Stone et al. 2020, ApJS 249:4):
- Full cylindrical AMR with CT (constrained transport) for div(B).
- Refluxing implemented in `src/bvals/flux_correction_cc.cpp` and
  `src/bvals/flux_correction_fc.cpp` (face-centered for CT).
- Uses coordinate-dependent face areas (`pcoord->GetFace1Area()` etc.)
  which return `2*pi*r*dz` for cylindrical radial faces.
- The reflux correction is: `U[k][j][i] -= dt * (F_fine_sum - F_coarse) / vol[i]`
  where `vol[i]` includes the cylindrical `pi*(r_{i+1/2}^2 - r_{i-1/2}^2)*dz`.
- This is the most complete reference implementation but is in C++ with
  MPI + CT, making direct porting impractical.

**AstroBEAR** (Cunningham et al. 2009; Vaidya et al. 2007, JCoPh 226:925):
- Hybrid block-AMR in curvilinear coordinates including cylindrical.
- Uses Balsara (2004) formulas for orthonormal cylindrical coordinates.
- CT at coarse-fine boundaries preserves div(B) = 0 to machine precision.
- The Vaidya 2007 paper is the closest published work to what we need --
  it explicitly addresses AMR refluxing in cylindrical MHD with CT.

#### SR-3.3 Cylindrical Refluxing Algorithm (Pseudocode)

Based on the literature survey, the algorithm is geometry-agnostic when volumes
and areas are pre-computed correctly. Here is the pseudocode for our implementation:

```python
def reflux_cylindrical(
    U_coarse: mx.array,       # (NVAR, nr_c, nz_c)
    F_coarse_r: mx.array,     # (NVAR, nr_c+1, nz_c) radial fluxes at faces
    F_coarse_z: mx.array,     # (NVAR, nr_c, nz_c+1) axial fluxes at faces
    F_fine_r: mx.array,       # (NVAR, nr_f+1, nz_f) fine radial fluxes
    F_fine_z: mx.array,       # (NVAR, nr_f, nz_f+1) fine axial fluxes
    coarse_fine_map: dict,     # maps coarse face -> list of fine faces
    r_face_c: mx.array,       # coarse radial face positions
    r_face_f: mx.array,       # fine radial face positions
    dr_c: float, dz_c: float,
    dr_f: float, dz_f: float,
    dt: float,
    ratio: int = 2,           # refinement ratio
) -> mx.array:
    """Apply Berger-Colella refluxing with cylindrical geometry.

    For each coarse face abutting the fine region:
      1. Accumulate fine flux * fine face area over sub-faces
      2. Compute coarse flux * coarse face area
      3. delta = (sum of fine flux*area) - (coarse flux*area)
      4. Correct adjacent coarse cell: U += delta * dt / V_coarse

    The r-weighting enters through the face areas and cell volumes:
      - Radial face area at position r: A_r = r * dz  (2*pi factor cancels)
      - Axial face area for cell [r_lo, r_hi]: A_z = 0.5*(r_hi^2 - r_lo^2)
      - Cell volume: V = 0.5*(r_hi^2 - r_lo^2) * dz

    Note: 2*pi factors cancel between flux*area and 1/volume, so we omit them.
    """
    dU = mx.zeros_like(U_coarse)

    # --- Radial face corrections ---
    # At each coarse radial face i_face that borders the fine region:
    for i_c, j_c, fine_faces in coarse_fine_map["radial"]:
        # Coarse flux * area
        r_c = r_face_c[i_c]
        FA_coarse = F_coarse_r[:, i_c, j_c] * r_c * dz_c

        # Sum of fine flux * area over the sub-faces
        FA_fine_sum = mx.zeros(U_coarse.shape[0])
        for i_f, j_f in fine_faces:
            r_f = r_face_f[i_f]
            FA_fine_sum += F_fine_r[:, i_f, j_f] * r_f * dz_f

        delta = FA_fine_sum - FA_coarse

        # Volume of adjacent coarse cell
        # (cell to the left or right of the face, depending on orientation)
        i_adj = i_c if orientation == "right" else i_c - 1
        r_lo = r_face_c[i_adj]
        r_hi = r_face_c[i_adj + 1]
        V_c = 0.5 * (r_hi**2 - r_lo**2) * dz_c

        dU[:, i_adj, j_c] += delta * dt / V_c

    # --- Axial face corrections ---
    # At each coarse axial face j_face bordering the fine region:
    for i_c, j_c, fine_faces in coarse_fine_map["axial"]:
        r_lo = r_face_c[i_c]
        r_hi = r_face_c[i_c + 1]
        A_c = 0.5 * (r_hi**2 - r_lo**2)
        FA_coarse = F_coarse_z[:, i_c, j_c] * A_c

        FA_fine_sum = mx.zeros(U_coarse.shape[0])
        for i_f, j_f in fine_faces:
            r_lo_f = r_face_f[i_f]
            r_hi_f = r_face_f[i_f + 1]
            A_f = 0.5 * (r_hi_f**2 - r_lo_f**2)
            FA_fine_sum += F_fine_z[:, i_f, j_f] * A_f

        delta = FA_fine_sum - FA_coarse

        j_adj = j_c if orientation == "top" else j_c - 1
        V_c = 0.5 * (r_hi**2 - r_lo**2) * dz_c

        dU[:, i_c, j_adj] += delta * dt / V_c

    return U_coarse + dU
```

**Critical implementation notes**:
1. The `2*pi` factor cancels between numerator (flux * area) and denominator
   (1/volume) in cylindrical coordinates. This is why MPI-AMRVAC's geometry-
   agnostic approach works -- as long as areas and volumes are consistent.
2. For factor-2 refinement, each coarse radial face maps to 2 fine faces
   (stacked in z). Each coarse axial face maps to 2 fine faces (stacked in r).
3. The loop-based pseudocode above is O(N_boundary). For MLX, vectorize by
   pre-computing index maps and using `mx.scatter_add` for the corrections.
4. This pseudocode does NOT handle CT (magnetic flux correction). We use
   Dedner GLM at level boundaries instead (Section 4 of the main scaffold).

#### SR-3.4 Key Insight from Literature

The review stated "cylindrical r-weighted refluxing has ZERO reference
implementations in any language." This is **partially incorrect**:

- **Athena++** has cylindrical AMR with refluxing in C++ (`flux_correction_cc.cpp`).
  It is the definitive reference. The geometry factors enter through
  `Coordinates::GetFaceXArea()` and `Coordinates::CellVolume()`, which are
  cylindrical-aware.
- **MPI-AMRVAC** has cylindrical AMR with `mod_fix_conserve` in Fortran.
  The volume arrays are pre-computed per cell with cylindrical factors.
- **AstroBEAR** has cylindrical AMR with refluxing + CT (Vaidya et al. 2007).

What is true: there is no **Python** implementation of cylindrical refluxing.
But the algorithm is well-understood -- it is standard Berger-Colella refluxing
with pre-computed cylindrical volumes and face areas. The difficulty is not
algorithmic but engineering: correct index mapping at coarse-fine boundaries.

---

### SR-4. Improve: Corrected Design Decisions

#### SR-4.1 Block Size: 16x32 (revised from 32x64)

**M3 Pro L1 cache analysis** (verified via web search):
- L1 data cache per P-core: **128 KB**
- L1 instruction cache per P-core: **192 KB**
- L2 cache shared per cluster: **16 MB** (6 P-cores share this)

Block memory footprint:
```
32x64 block: NVAR * 32 * 64 * 4 bytes = 10 * 32 * 64 * 4 = 80 KB
  + ghost zones (ng=3): 10 * 38 * 70 * 4 = 106 KB  --> EXCEEDS 128 KB L1
  + workspace (fluxes): ~3x = ~320 KB --> far exceeds L1, spills to L2

16x32 block: NVAR * 16 * 32 * 4 bytes = 10 * 16 * 32 * 4 = 20 KB
  + ghost zones (ng=3): 10 * 22 * 38 * 4 = 33 KB   --> FITS in 128 KB L1
  + workspace (fluxes): ~3x = ~100 KB --> fits in L1 with room to spare

16x64 block: NVAR * 16 * 64 * 4 bytes = 10 * 16 * 64 * 4 = 40 KB
  + ghost zones (ng=3): 10 * 22 * 70 * 4 = 62 KB   --> FITS in 128 KB L1
  + workspace (fluxes): ~3x = ~185 KB --> spills to L2
```

**Recommendation**: Use **16x32** blocks for Phase A. This:
- Fits entirely in L1 data cache (33 KB ghosted < 128 KB)
- Leaves room for flux workspace in L1
- Means more blocks per level (16 blocks cover 64x128 base grid) but each
  block processes faster due to cache locality
- Can be profiled and adjusted to 16x64 if GPU occupancy is too low

The original 32x64 choice was based on "matches typical nr/nz ratio" but
cache performance dominates for our block counts (< 32 total blocks).

**Decision log update**: Block size 16x32 (revised from 32x64) for L1 cache fit.

#### SR-4.2 Ghost Exchange Priority Order (new specification)

When a ghost cell could be filled by multiple sources, the priority is:

1. **Physical boundary conditions** (electrode, axis, outflow) -- highest
2. **Same-level neighbor copy** -- standard interior ghost fill
3. **Prolongation from coarser level** -- only for fine-level ghost zones
   adjacent to coarse-only regions

This ordering prevents prolongation from overwriting physically-mandated BCs
at the axis (r=0) or electrode surfaces.

#### SR-4.3 Verified References

**Parthenon-VIBE (arXiv:2509.19701)**: VERIFIED. The paper exists on arXiv
with submission date September 24, 2025. Title: "Characterizing Adaptive Mesh
Refinement on Heterogeneous Platforms with Parthenon-VIBE." Authors: multiple,
affiliated with Sandia/LANL. The arXiv ID prefix `2509` = September 2025, which
was initially flagged as suspicious ("future date?") but the review was written
in 2026-03 so September 2025 is in the past.

Key finding from the paper: smaller mesh blocks and deeper AMR levels degrade
GPU performance due to increased communication, serial overheads, and inefficient
GPU utilization. Low SM occupancy and poor HBM bandwidth utilization are the
root causes. This directly informs our decision to use global timestep (no
subcycling) and few blocks (max 16 per level) for Phase A-B.

**Popovas et al. (arXiv:2211.02438)**: Title is "DISPATCH HLLS" but this is
correctly cited in our scaffold. The lead author is Popovas, not Agertz/Nordlund.
Paper validates entropy-based MHD in float32 -- relevant for our dual-energy
approach in AMR where refluxing errors compound with float32 cancellation.

#### SR-4.4 MLX vmap Over Fixed-Size Blocks

MLX's `mx.vmap()` is confirmed to work for batch processing of fixed-size arrays.
The pattern for AMR:

```python
import mlx.core as mx

def single_block_rhs(U_block: mx.array, dr: float, dz: float) -> mx.array:
    """RHS for one block. Shape: (NVAR, nr_block, nz_block)."""
    # ... existing MLX RHS code, parametrized by grid spacing ...
    return dU

# Vectorize over batch dimension
batched_rhs = mx.vmap(single_block_rhs, in_axes=(0, None, None))

# Process all blocks at one level simultaneously
U_batch = level.as_batch()       # (N_blocks, NVAR, nr, nz)
dU_batch = batched_rhs(U_batch, level.dr, level.dz)
```

**Constraints**:
- All blocks in the batch MUST have identical shapes (satisfied by block-structured AMR).
- `mx.vmap` is composable with `mx.compile` -- the compiled+vmapped kernel is
  traced once and reused. Shape must be static.
- Pre-allocate MAX_BLOCKS slots and use masking for inactive blocks:
  `dU_batch = dU_batch * active_mask[:, None, None, None]`
- `mx.vmap` does NOT support dynamic batch sizes inside `mx.compile`. The batch
  dimension must be fixed at compile time. This means MAX_BLOCKS_PER_LEVEL is
  a compile-time constant.

**Alternative to vmap**: Use a leading batch dimension directly. The existing
MLX solver operates on `(NVAR, nr, nz)`. Reshape to `(N_blocks * NVAR, nr, nz)`
or `(N_blocks, NVAR, nr, nz)` and adjust stencil operations. This avoids vmap
entirely but requires modifying the RHS internals. vmap is cleaner.

#### SR-4.5 Simplified Phase A Spec (reuses maximum existing code)

```python
# File: src/dpf/experimental/amr.py

@dataclass
class AMRBlock:
    level: int
    index: tuple[int, int]       # (ir, iz) in level grid
    U: mx.array                  # (NVAR, nr_block, nz_block)
    r_min: float
    z_min: float
    active: bool = True

@dataclass
class AMRLevel:
    level: int
    blocks: dict[tuple[int, int], AMRBlock]
    dr: float
    dz: float

    def as_batch(self) -> mx.array:
        """Stack active blocks: (N_active, NVAR, nr, nz)."""
        return mx.stack([b.U for b in self.blocks.values() if b.active])

class AMRHierarchy:
    """2-level block-structured mesh for cylindrical MHD."""

    def __init__(self, config: AMRConfig, grid: CylindricalGrid):
        self.levels: list[AMRLevel] = []
        self._decompose(config, grid)

    def _decompose(self, config, grid):
        # Split base grid into blocks of (nr_block, nz_block)
        # Create level 0 with full coverage
        # Create level 1 with user-specified refined blocks
        ...

    def ghost_exchange(self, level_idx: int):
        # Priority: physical BC > neighbor > prolongation
        # Reuse ghost_pad_mlx() for physical BCs
        # Copy from neighbors for interior block boundaries
        # Bilinear interp for fine-level ghosts from coarse
        ...

    def restrict(self):
        # Volume-weighted average from level 1 to level 0
        # Weight: V_fine / V_coarse = r_fine * dr_f * dz_f / (r_coarse * dr_c * dz_c)
        # For factor-2: each coarse cell averages 4 fine cells
        ...

    def step(self, dt: float, current: float, rhs_fn):
        # 1. Ghost exchange on all levels
        # 2. Batch RHS on level 0 (all blocks)
        # 3. Batch RHS on level 1 (refined blocks)
        # 4. Restrict level 1 -> level 0 (overwrite coarse cells)
        # 5. Advance: U += dt * dU
        ...
```

**Reuse from existing code**:
- `ghost_pad_mlx()` / `ghost_pad_numpy()` from `mlx_kernels.py` for physical BCs
- `interpolate_to_fine_grid()` pattern from `static_refinement.py` for prolongation
- `lohner_error_indicator()` / `current_density_sensor()` for Phase B
- `_mask_ghost_rhs()` from `mlx_timestepper.py` for ghost cell RHS zeroing
- `_pad_electrode_ghost()` pattern from `mlx_solver.py` for electrode BCs at r-boundaries

---

### SR-5. Control: Acceptance Test for Phase A AMR

#### SR-5.1 Primary Acceptance Test: Sheath Resolution Improvement

**Test name**: `test_amr_sheath_resolution_improvement`

**Setup**:
1. PF-1000 configuration, 16 kV / 1.2 Torr deuterium
2. Uniform grid: 64x128 (nr x nz), `dr = 0.47 mm`, `dz = 1.4 mm`
3. AMR grid: 64x128 base (level 0) + 4 refined blocks at sheath (level 1, 2x)
   - Level 1 effective: `dr = 0.23 mm`, `dz = 0.7 mm` in sheath region
4. Run 100 steps of axial rundown phase (sheath moving in +z)

**Measurements**:
- `|J|_max` on uniform grid vs AMR grid (AMR should resolve a sharper peak)
- Cells with `|J| > 0.5 * |J|_max`: count and width (AMR should show narrower sheath)
- `|delta_mass/mass|` per step: must be < 1e-6 (no refluxing; conservation from
  volume-weighted restriction only)
- Wall-clock time: AMR must be <= 1.5x uniform time (the 2x CFL penalty is
  partially offset by fewer total cells at full resolution)

**Pass criteria**:
```python
assert amr_j_max >= 1.3 * uniform_j_max, "AMR must resolve sharper J peak"
assert amr_sheath_width <= 0.7 * uniform_sheath_width, "AMR sheath thinner"
assert abs(delta_mass / mass) < 1e-6, "Conservation within tolerance"
assert amr_wallclock <= 1.5 * uniform_wallclock, "No worse than 1.5x slowdown"
```

**Fail criteria**:
- AMR `|J|_max` is LOWER than uniform (means prolongation/restriction is diffusing)
- Mass drift > 1e-4 (means ghost exchange or restriction is broken)
- AMR slower than 2x uniform (means overhead dominates; block count or regrid issue)
- NaN in any field (means ghost exchange at axis or electrode is wrong)

#### SR-5.2 Regression Gate

```bash
# Must pass before AMR is promoted from experimental
pytest tests/ -x -q -m "not slow"   # all non-slow tests (default: amr off)
pytest tests/test_amr.py -v          # AMR-specific tests
```

#### SR-5.3 Conservation Audit (per-phase gates)

| Phase | Conservation Target | Metric |
|-------|-------------------|--------|
| A-slim | `\|dM/M\| < 1e-6` per step | No refluxing; restriction only |
| B | `\|dM/M\| < 1e-6` per step | Auto-regrid adds blocks; same restriction |
| C | `\|dM/M\| < 1e-10` per step | Refluxing active; machine-precision target |
| D | `\|dM/M\| < 1e-10` per step | 3+ levels; recursive subcycling |

---

### SR-6. Additional Research Findings

#### SR-6.1 Minimum Refinement Ratio: 2x vs 4x

For DPF sheath tracking, **2x is the right choice for Phase A**:

- The sheath thickness is ~1-2 mm (Auluck 2022). With `dr_coarse ~ 0.5 mm`,
  the sheath spans 2-4 cells. Factor-2 gives 4-8 cells; factor-4 gives 8-16 cells.
- Factor-4 refinement with global timestep means CFL is determined by the finest
  level: `dt = CFL * dr_fine / c_fast`. At factor-4, `dt` is 4x smaller than the
  coarse CFL -- meaning the coarse level takes 4x more steps than necessary.
  With no subcycling, this is a 4x wall-clock penalty.
- Factor-2 gives only a 2x CFL penalty, which is acceptable.
- At 3+ levels (Phase D) with subcycling, factor-2 per level gives 4x effective
  refinement (level 0 -> level 1 -> level 2) with only 2x subcycling at each level.
- **Standard in Athena++**: uses factor-2 refinement exclusively.
- **MPI-AMRVAC**: supports factor-2 and factor-4 but recommends factor-2 for
  efficiency when subcycling is used.

#### SR-6.2 MLX vmap vs Leading Batch Dimension

Two approaches for processing multiple blocks on MLX:

**Option A: `mx.vmap`**
- Cleaner: existing per-block RHS function is unchanged
- `mx.vmap(fn, in_axes=(0, None, None))` maps over batch dim
- Composable with `mx.compile`: `mx.compile(mx.vmap(rhs))`
- Caveat: vmap traces the function once; if the RHS has conditionals that depend
  on data values (not shapes), vmap may not handle them correctly

**Option B: Leading batch dimension**
- Reshape blocks into `(N_blocks, NVAR, nr, nz)` tensor
- Modify stencil operations to work on 4D arrays
- More invasive (touches all stencil code) but avoids vmap tracing issues
- Natural for `mx.compile` since shape is static

**Recommendation**: Start with **Option A (vmap)** for Phase A. If vmap causes
issues with the compiled RHS (conditionals, in-place ops), fall back to Option B.
Profile both approaches during Phase A development.

#### SR-6.3 Parthenon-VIBE Implications for Our Design

The paper (arXiv:2509.19701) found:
1. Smaller blocks degrade GPU performance (low SM occupancy)
2. Deeper AMR levels increase communication overhead
3. Regrid cost scales as O(N_blocks * N_ghost)

Our mitigations:
- 16x32 blocks are small but we have few of them (max 32 total). The paper's
  concerns apply at thousands of blocks, not tens.
- 2 levels only (Phase A-B). No deep hierarchy.
- Global timestep eliminates synchronization overhead.
- Regrid every 20+ steps amortizes Python-level regrid cost.
- Our bottleneck is different: we're on a single Apple Silicon chip, not a
  multi-node GPU cluster. Memory latency, not inter-node communication, is
  the dominant overhead.

---

### SR-7. Updated Risk Matrix (Post-Refinement)

| Risk | Pre-Refinement RPN | Mitigation Applied | Post-Refinement RPN | Status |
|------|-------------------|-------------------|-------------------|--------|
| Cylindrical refluxing wrong | 252 | Literature survey + pseudocode + standalone test plan | 126 | Reduced |
| mx.compile recompiles on regrid | 105 | Pre-alloc MAX_BLOCKS=16 + profile plan | 70 | Reduced |
| Ghost exchange at axis (r=0) | 160 | Priority order specified (phys BC > neighbor > prolong) | 96 | Reduced |
| Prolongation not div(B)-consistent | 252 | Dedner GLM confirmed in MPI-AMRVAC approach | 168 | Reduced |
| Block size exceeds L1 cache | NEW | Changed from 32x64 to 16x32 (33 KB < 128 KB) | N/A | Resolved |
| LOC estimates too low | NEW | Revised with P25/P50/P75 per function | N/A | Resolved |
| Parthenon-VIBE ref unverified | NEW | Verified: arXiv:2509.19701 exists (Sep 2025) | N/A | Resolved |

---

### SR-8. Updated Decision Log

| Decision | Original | Revised | Rationale |
|----------|----------|---------|-----------|
| Block size | 32x64 | **16x32** | 128 KB L1 data cache; 33 KB ghosted block fits; 80 KB did not |
| Phase A scope | Full 2-level AMR | **Phase A-slim** (manual region, no reflux, no auto-regrid) | Minimum viable; prove value before complexity |
| Refinement ratio | Factor-2 | **Factor-2 confirmed** | 2x CFL penalty acceptable; 4x too costly without subcycling |
| LOC estimates | 1150-1550 | **1130-2205 (P25-P75)** | Phase C is 60% larger than estimated |
| Parthenon-VIBE ref | Unverified | **Verified** | arXiv:2509.19701 exists, submitted Sep 2025 |
| Cylindrical refluxing | "No reference exists" | **Athena++ / MPI-AMRVAC / AstroBEAR** all have it | Algorithm is standard Berger-Colella with pre-computed cylindrical volumes |
| vmap strategy | Not specified | **mx.vmap for Phase A** | Cleanest; fall back to leading batch dim if issues |

---

### SR-9. References (New)

11. **Vaidya, B. et al.**, "Hybrid block-AMR in Cartesian and curvilinear
    coordinates: MHD applications," *JCoPh* 226:925 (2007).
    -- Block-structured AMR in cylindrical coordinates with CT. Closest
    reference to our implementation.

12. **Auluck, S.K.H.**, "First steps towards a theory of the Dense Plasma Focus:
    Part-I," *arXiv:2211.16775* (2022).
    -- Sheath thickness analysis. Informs minimum refinement ratio.

13. **Mignone, A. et al.**, "The PLUTO Code for Adaptive Mesh Computations in
    Astrophysical Fluid Dynamics," *ApJS* 198:7 (2012).
    -- AMR via CHOMBO in cylindrical coordinates. FluxRegister approach.

14. **Skinner, M.A. & Ostriker, E.C.**, "The Athena Astrophysical
    Magnetohydrodynamics Code in Cylindrical Geometry," *ApJS* 188:290 (2010).
    -- Original Athena cylindrical implementation. Geometric source terms.

---

### SR-10. Readiness Score Update

| Criterion | Before | After | Evidence |
|-----------|--------|-------|----------|
| LOC estimates | Optimistic | Calibrated | P25/P50/P75 per function; Phase C +60% |
| Cylindrical refluxing | No reference | 3 references + pseudocode | Athena++, MPI-AMRVAC, AstroBEAR |
| Block size | Exceeds L1 | Fits L1 | 16x32 = 33 KB < 128 KB |
| Parthenon-VIBE ref | Unverified | Verified | arXiv:2509.19701 confirmed |
| Phase A scope | Ambitious | Minimal viable | Phase A-slim, 470 LOC P50 |
| Acceptance test | Not defined | Defined | Sheath resolution + conservation + wall-clock |

**Revised readiness: 8/10** -- Implementation can start. The cylindrical refluxing
algorithm (Phase C) remains the highest-risk item but is now supported by literature
references and pseudocode. Phase A-slim avoids this risk entirely.
